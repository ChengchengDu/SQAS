import os
import logging
from tqdm import tqdm, trange
import math
import random
import sys
sys.path.append("/home/jazhan/code/QaEnDeBart/")
from train_utils.log import init_logger, logger
from .integrations_ import init_deepspeed, is_fairscale_available

import numpy as np
import torch
import torch.distributed as dist
# from transformers.optimization import AdamW
from transformers import AdamW, PreTrainedModel
from transformers import get_polynomial_decay_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


if is_fairscale_available():
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


class Trainer:
    def __init__(
            self,
            args,
            train_dataloader,
            valid_dataloader,
            model):
        # 这个模型是没有进行任何的封装的  只是迁移到了device上面
        self.args = args

        self.model = model
        self.model_wrapped = model
        self.deepspeed = None
        if args.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(log_dir=args.log_dir)

        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        self.num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.total_num_update > 0:
            self.t_total = args.total_num_update
            length = len(train_dataloader)
            args.epochs = args.total_num_update // (length // args.gradient_accumulation_steps) + 1
            self.epochs = args.epochs
        else:
            length = len(train_dataloader)
            self.t_total = length // args.gradient_accumulation_steps * args.epochs
        # Prepare optimizer and schedule(polynomial_decay and warmup)

        self.sharded_dpp = False
        if args.sharded_ddp:
            if args.local_rank == -1:
                raise ValueError("Using sharded DDP only works in distributed training.")
            elif not is_fairscale_available():
                raise ImportError("Sharded DDP training requires fairscale: `pip install fairscale`.")
            elif args.deepspeed:
                raise ValueError("can't use --sharded_ddp together with --deepspeed.")
            else:
                self.sharded_dpp = True

        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]    # 优化器这一块需要和fairseq进行对比和修正

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_eps)
        self.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                          num_warmup_steps=args.warmup_steps,
                                                          num_training_steps=t_total)

        if os.path.isfile(os.path.join(args.load_checkpoint, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.load_checkpoint, "scheduler.pt")
        ):
            # load in optimizer and scheduler states

            self.optimizer.load_state_dict(torch.load(os.path.join(args.load_checkpoint, "optimizer.pt")))
            self.scheduler.load_state_dict(torch.load(os.path.join(args.load_checkpoint, "scheduler.pt")))

        if self.deepspeed:
            # Not sure how to check if there is a saved deepspeed checkpoint, but since it just return None if it fails to find a deepspeed checkpoint this is sort of a check-n-load function
            self.deepspeed.load_checkpoint(args.load_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True)


        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from http://wwww.github.com/nvidia/apex/ to use"
                                  "fp16 training")

            self.model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training(should be after apex fp16 initialization)
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(model)

        # Distributed training(should be after apex fp16 initialization)
        if args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,

            )
    def train(self):
        # training start
        logger.info("****** Running training ******")
        logger.info(" Num example = %d ", len(self.train_dataloader) * self.args.per_gpu_train_batch_size)
        logger.info(" Num Epoch = %d", self.args.epochs)
        logger.info(" Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)

        logger.info(
            " Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.per_gpu_train_batch_size * self.args.gradient_accumulation_steps
            * (dist.get_world_size() if self.args.local_rank != -1 else 1)
        )

        logger.info(" Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info(" Total optimizer steps = %d", self.t_total)

        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(
                self,
                num_training_steps=self.t_total
            )
            self.model = model.module
            self.model_wrapped = model  # will get further wrapped in DDP
            self.deepspeed = model  # DeepSpeedEngine object
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        model = self.model_wrapped
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.sharded_dpp:
            model = ShardedDDP(model, self.optimizer)
        elif self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )

        if model is not self.model:
            self.model_wrapped = model

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.args.output_dir):
            try:
                checkpoint_suffix = self.args.load_checkpoint.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (self.length // self.args.gradient_accumulation_steps)

                logger.info(" continue training from checkpoint, will skip to save global_steps")
                logger.info(" Continue training from epoch %d", epochs_trained)
                logger.info(" Continue training from global steps %d", global_step)
                logger.info(" Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info(" Starting fine-tuning")

        total_non_pad_words = 0   # 总的单词数
        total_correct_words = 0
        cur_correct_num = 0  # 一个累积梯度的ppl
        cur_non_pad_num = 0  # 一个累积梯度的准确率
        cur_sum_loss = 0
        tr_ppl, logging_ppl = 0, 0
        tr_acc, logging_acc = 0, 0
        tr_step = 0
        tr_loss, logging_loss = 0.0, 0.0  # 全局的损失和日志的损失
        tr_summary_loss, logging_summary_loss = 0.0, 0.0
        tr_qa_loss, logging_qa_loss = 0.0, 0.0
        tr_mim_loss, logging_mim_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(self.args.epochs), desc="Epochs", disable=args.local_rank not in [-1, 0]
        )
        # Added here for reproductibility
        set_seed(self.args)

        # 整个模型包括两个任务 encoder端使用BCE损失的问答任务   decoder端的摘要任务  (encoder-decoder端的互信息任务)
        best_valid_sum_loss_1 = float("inf")
        best_valid_loss_1 = float("inf")

        best_valid_sum_loss_2 = float("inf")
        best_valid_loss_2 = float("inf")

        best_valid_sum_loss_3 = float("inf")
        best_valid_loss_3 = float("inf")

        best_ppl = float("inf")
        best_acc = float("inf")
        oom_time = 0    # 记录out of memory 的次数
        for epoch in tqdm(train_iterator, desc="Epoch"):
            epoch_iterator = tqdm(self.train_dataloader, desc="Iterator", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                tr_step += 1
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= -1
                    continue

                self.model.train()

                #
                # for name, parms in model.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data))
                #     print("-->grad_value", torch.mean(parms.grad))

                #

                # 模型的输入这一块
                source_ids = batch["input_ids"].long().to(args.device)
                attention_mask = batch["attention_mask"].long().to(args.device)
                summary_labels = batch["target_ids"].long().to(args.device)
                start_positions = batch["start_positions"].long().to(args.device)     # 注意这个改成了实际位置作为最终的labels
                end_positions = batch["end_positions"].long().to(args.device)

                inputs = {
                    "input_ids": source_ids,
                    "attention_mask": attention_mask,
                    "labels": summary_labels,
                    "start_positions": start_positions,
                    "end_positions": end_positions,
                    "return_dict": True
                }

                try:
                    outputs = self.model(**inputs)
                    # tb_writer.add_graph(model, source_ids)
                    loss, qa_loss, sum_loss = outputs.loss, outputs.qa_loss, outputs.summary_loss
                    loss = outputs.loss
                    sum_loss = outputs.summary_loss
                    mim_loss = outputs.mim_loss

                    correct_nums, non_pad_tokens = stats(
                        outputs.logits, inputs["labels"], args.pad_token_id)

                    cur_correct_num += correct_nums
                    cur_non_pad_num += non_pad_tokens
                    total_non_pad_words += non_pad_tokens
                    total_correct_words += correct_nums
                    cur_sum_loss += sum_loss


                    if self.args.n_gpu > 1:
                        loss.mean()

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    if self.args.fp16:
                        if self.args.fp16:
                            try:
                                from apex import amp
                            except ImportError:
                                raise ImportError("Please install apex from http://wwww.github.com/nvidia/apex/ to use"
                                                  "fp16 training")
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    #
                    # for name, parms in model.named_parameters():
                    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data))
                    #     print("-->grad_value", torch.mean(parms.grad))

                    tr_loss += loss.item()
                    tr_qa_loss += qa_loss.item()
                    tr_mim_loss += mim_loss.item()
                    tr_summary_loss += sum_loss.item()

                    cur_sum_loss += sum_loss

                    logger.info(
                        "epoch:{}, global_step:{}, loss:{}, qa_loss:{}, sum_loss:{}, mim_loss: {}".format(
                            epoch, global_step, loss, qa_loss, sum_loss, mim_loss
                        )
                    )

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        cur_acc = accuracy(cur_correct_num, cur_non_pad_num)
                        total_acc = accuracy(total_correct_words, total_non_pad_words)
                        cur_correct_num = 0
                        cur_non_pad_num = 0

                        cur_ppl = cal_ppl(cur_sum_loss / self.args.gradient_accumulation_steps)

                        cur_sum_loss = 0
                        total_ppl = cal_ppl(tr_summary_loss / global_step / self.args.gradient_accumulation_steps)   # 计算有问题
                        logger.info(
                            f"global_step:{global_step}, cur_acc:{cur_acc}, total_acc:{total_acc}, "
                            f"cur_ppl:{cur_ppl}, total_ppl:{total_ppl}")

                        tr_acc += total_acc
                        tr_ppl += total_ppl
                        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                            if self.args.fp16:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)

                            if hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                        self.model.zero_grad()
                        global_step += 1

                        # Log Metrics
                        if self.args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                            self.tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                            self.tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)  # 日志步数之内的平均损失
                            self.tb_writer.add_scalar(
                                "summary_loss", (tr_summary_loss - logging_summary_loss) / args.logging_steps, global_step)
                            self.tb_writer.add_scalar(
                                "qa_loss", (tr_qa_loss - logging_qa_loss) / args.logging_steps, global_step
                            )
                            self.tb_writer.add_scalar(
                                "mim_loss", (tr_mim_loss - logging_mim_loss) / args.logging_steps, global_step
                            )
                            self.tb_writer.add_scalar(
                                "total_acc", (tr_acc - logging_acc) / args.logging_steps, global_step
                            )
                            self.tb_writer.add_scalar(
                                "total_ppl", (tr_ppl - logging_ppl) / args.logging_steps, global_step
                            )

                            logging_loss = tr_loss
                            logging_summary_loss = tr_summary_loss
                            logging_qa_loss = tr_qa_loss
                            logging_mim_loss = tr_mim_loss
                            logging_acc = tr_acc
                            logging_ppl = tr_ppl

                        if self.args.local_rank in [-1, 0] and self.args.save_steps > 0 and global_step % args.save_steps == 0:
                            logger.info('Saving model at global_steps{}'.format(global_step))
                            save_path = os.path.join(self.args.output_dir,  "checkpoint-" + str(global_step))
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)

                            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                            model_to_save.save_pretrained(save_path)
                            logger.info("Saving model checkpoint to %s", save_path)

                            
                            if self.deepspeed:
                                self.deepspeed.save_checkpoint(save_path)
                                # Save optimizer and scheduler
                            if self.sharded_dpp:
                                self.optimizer.consolidate_state_dict()
                            if not self.deepspeed:
                                torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
                                torch.save(self.scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", save_path)
                            valid_loss, valid_sum_loss, valid_qa_loss, valid_mim_loss, valid_acc, valid_ppl = validation(
                                self.valid_dataloader, self.model, self.args)

                            if best_ppl > valid_ppl:
                                logger.info('Saving best valid ppl model at global_steps{}'.format(global_step))
                                best_valid_ppl_dir = os.path.join(args.output_dir, "best_ppl")
                                if not os.path.exists(best_valid_ppl_dir):
                                    os.makedirs(best_valid_ppl_dir)

                                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                                model_to_save.save_pretrained(best_valid_ppl_dir)
                                logger.info("Saving model checkpoint to %s", best_valid_ppl_dir)
                                
                                torch.save(self.optimizer.state_dict(), os.path.join(best_valid_ppl_dir, "optimizer.pt"))
                                torch.save(self.scheduler.state_dict(), os.path.join(best_valid_ppl_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", best_valid_ppl_dir)
                                if self.deepspeed:
                                    self.deepspeed.save_checkpoint(best_valid_ppl_dir)
                                if self.sharded_dpp:
                                    self.optimizer.consolidate_state_dict()
                                if not self.deepspeed:
                                    torch.save(self.optimizer.state_dict(), os.path.join(best_valid_ppl_dir, "optimizer.pt"))
                                    torch.save(self.scheduler.state_dict(), os.path.join(best_valid_ppl_dir, "scheduler.pt"))
                                    logger.info("Saving optimizer and scheduler states to %s", save_path)

                            if best_acc > valid_acc:
                                logger.info('Saving best valid acc model at global_steps{}'.format(global_step))
                                best_valid_acc_dir = os.path.join(args.output_dir, "best_acc")
                                if not os.path.exists(best_valid_acc_dir):
                                    os.makedirs(best_valid_acc_dir)

                                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                                model_to_save.save_pretrained(best_valid_acc_dir)
                                logger.info("Saving model checkpoint to %s", best_valid_acc_dir)
                                if self.deepspeed:
                                    self.deepspeed.save_checkpoint(best_valid_acc_dir)
                                if self.sharded_dpp:
                                    self.optimizer.consolidate_state_dict()
                                if not self.deepspeed:
                                    torch.save(self.optimizer.state_dict(), os.path.join(best_valid_acc_dir, "optimizer.pt"))
                                    torch.save(self.scheduler.state_dict(), os.path.join(best_valid_acc_dir, "scheduler.pt"))
                                    logger.info("Saving optimizer and scheduler states to %s", save_path)

                            # 保存总体的损失
                            save_index = 0
                            if valid_loss < best_valid_loss_1:
                                best_valid_loss_1 = valid_loss
                                save_index = 1
                            elif valid_loss < best_valid_loss_2:
                                best_valid_loss_2 = valid_loss
                                save_index = 2
                            elif valid_loss < best_valid_loss_3:
                                best_valid_loss_3 = valid_loss
                                save_index = 3
                            else:
                                save_index = -1

                            if save_index != -1:
                                logger.info('Saving best valid loss model at global_steps{}'.format(global_step))
                                best_valid_loss_dir = os.path.join(args.output_dir, "best_loss_" + str(save_index))
                                if not os.path.exists(best_valid_loss_dir):
                                    os.makedirs(best_valid_loss_dir)

                                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                                model_to_save.save_pretrained(best_valid_loss_dir)
                                logger.info("Saving model checkpoint to %s", best_valid_loss_dir)

                                torch.save(self.optimizer.state_dict(), os.path.join(best_valid_loss_dir, "optimizer.pt"))
                                torch.save(self.scheduler.state_dict(), os.path.join(best_valid_loss_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", best_valid_loss_dir)

                                if self.deepspeed:
                                    self.deepspeed.save_checkpoint(best_valid_loss_dir)
                                if self.sharded_dpp:
                                    self.optimizer.consolidate_state_dict()
                                if not self.deepspeed:
                                    torch.save(self.optimizer.state_dict(),
                                               os.path.join(best_valid_loss_dir, "optimizer.pt"))
                                    torch.save(self.scheduler.state_dict(),
                                               os.path.join(best_valid_loss_dir, "scheduler.pt"))
                                    logger.info("Saving optimizer and scheduler states to %s", save_path)

                    if (self.args.total_num_update > 0) and (global_step > self.args.total_num_update):
                        epoch_iterator.close()
                        break

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        logger.info("WARNING: ran out of memory, times:{}".format(oom_time))
                        if hasattr(torch.cuda, "empty_cache"):
                            torch.cuda.empty_cache()
                    else:
                        logger.info(str(exception))
                        raise exception
            if (self.args.total_num_update > 0) and (global_step > self.args.total_num_update):
                epoch_iterator.close()
                break

        if self.args.local_rank in [-1, 0]:
            self.tb_writer.close()
        logger.info(
            "training has done!"
        )

        return global_step, tr_loss


def cal_ppl(loss):
    """ compute perplexity """
    return math.exp(min(loss,  100))


def accuracy(n_correct, n_words):
    """ compute accuracy 总的正确数 / 总的单词数"""
    return 100 * (n_correct / n_words)


def stats(scores, target, padding_idx, global_steps=0):
    pred = scores.max(-1)[1]
    non_padding = target.ne(padding_idx)
    num_correct = pred.eq(target) \
        .masked_select(non_padding) \
        .sum() \
        .item()

    num_non_padding = non_padding.sum().item()

    return num_correct, num_non_padding