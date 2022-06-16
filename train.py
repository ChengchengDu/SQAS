import argparse
import logging
import os
import random
import numpy as np

import torch
import torch.distributed as dist
from transformers.configuration_bart import BartConfig
from transformers.tokenization_bart import BartTokenizer
import sys
sys.path.append("/home/jazhan/code/query_based_summarization/")

# from data.data_loader import get_SummarySquad_dataloader
from train_utils.trainer import trainer
from train_utils.log import logger, init_logger
from data_utils.data_loader import DataTrainer
from model.en_de_sumqa import EncDecQaMimBart
import multiprocessing

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '--max_tokens', defalut=2048, type=int, help="the max tokens for input ids length"
    # )   fairseq中的max_tokens指的是一个batch中的最大token个数
    # 我们使用batch_size进行代替

    parser.add_argument(
        '--layernorm_embedding', action="store_true"
    )

    parser.add_argument(
        '--share_all_embedding', action="store_true"
    )

    parser.add_argument(
        '--share_decoder_input_output_embed', action="store_true"
    )

    parser.add_argument(
        '--label_smoothing', default=0.1, help="label smoothing", type=float
    )

    parser.add_argument(
        '--dropout', default=0.1, type=float
    )

    parser.add_argument(
        '--attention_dropout', default=0.1, type=float
    )

    parser.add_argument(
        '--weight_decay', default=0.01, type=float
    )

    parser.add_argument(
        '--optimizer', default="adamW"
    )

    parser.add_argument(
        '--adam_betas', default=(0.9, 0.999), type=tuple
    )

    parser.add_argument(
        '--adam_eps', default=1e-8, type=float
    )

    parser.add_argument(
        '--max_grad_norm', default=0.1, type=float, help="clip threshold of gradients"
    )

    parser.add_argument(
        '--lr_schedule',  default="polynomial_decay", type=str
    )

    parser.add_argument(
        '--learning_rate', default=3e-5, type=float
    )

    parser.add_argument(
        '--total_num_update', default=20000, type=int
    )

    parser.add_argument(
        '--warmup_steps', default=500, type=int
    )

    parser.add_argument(
        '--fp16', action="store_true", help="whether to use 16-bit precision instead of 32-bit"
    )

    parser.add_argument(
        '--fp16_opt_level', type=str, default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    )

    parser.add_argument(
        '--sum_features_data_dir', default="/", type=str, help="the cached dir of summary squad features"
    )

    parser.add_argument(
        '--logging_steps', type=int, default=1000
    )

    parser.add_argument(
        '--save_steps', default=1000, type=int
    )

    parser.add_argument(
        '--seed', type=int, default=42
    )

    parser.add_argument(
        '--evaluate_during_training', action="store_true", help="Run evaluation during training at each logging steps"
    )

    parser.add_argument(
        '--debug', action="store_true"
    )

    parser.add_argument(
        '--overwrite_output_dir', action="store_true", help="whether over write output dir to save checkpoint "
    )

    parser.add_argument(
        '--output_dir', default=None, type=str, required=True,
        help="the output of directory where the model checkpoints and predictions will be written"
    )

    parser.add_argument(
        '--local_rank', default=-1, type=int, help="local_rank for distributed training on gpus"
    )

    parser.add_argument(
        '--no_cuda', action="store_true", help="Whether not use CUDA when available"
    )

    parser.add_argument(
        '--model_name_or_path', default="facebook/bart-large", type=str, help="pretrained model for load"
    )

    parser.add_argument(
        '--cache_dir', default=None, type=str, help="the directory of saving pretrained model "
    )

    parser.add_argument(
        '--per_gpu_train_batch_size', default=8, type=int, help="batch size for training and prediction"
    )

    parser.add_argument(
        '--eval_batch_size', default=8, type=int
    )

    parser.add_argument(
        '--gradient_accumulation_steps', default=1, type=int,
        help="gradient steps for training,for cnndm, one node update_freq is 4, for xsum update_freq is equal to 2"
    )

    parser.add_argument(
        '--do_train', action="store_true", help="whether training"
    )

    parser.add_argument(
        '--add_final_layer_norm', action="store_true"
    )

    parser.add_argument(
        '--epochs', type=int, default=3
    )

    parser.add_argument(
        '--sum_train_features_file_name', default="cached-train-cnn-features-512", type=str,
        help="the train file name of summary features "
    )
    parser.add_argument(
        "--sum_valid_features_file_name", default="cached-dev-cnn-features-512", type=str,
        help="the valid file name of summary features"
    )
    parser.add_argument(
        "--load_checkpoint", default="./checkpoint/", type=str,
        help="has trained checkpoint to init bart for encoder-decoder summary"
    )
    parser.add_argument(
        "--log_file", default="./train.log", type=str,
        help="the file for write log"
    )
    parser.add_argument(
        "--log_dir", required=False, type=str,
    )
    parser.add_argument(
        "--alpha_qa", type=float, default=1.0, help="qa_loss scale", required=False
    )
    parser.add_argument(
        "--beta_sum", type=float, default=1.0, help="sum_loss scale", required=False
    )
    parser.add_argument(
        "--beta_mim_en", type=float, default=1.0, help="en_mim_loss scale", required=False
    )
    parser.add_argument(
        "--beta_mim_de", type=float, default=1.0, help="de_mim_loss scale", required=False
    )
    parser.add_argument(
        "--only_qa", action="store_true", help="training for qa and summary"
    )
    parser.add_argument(
        "--only_mim", action="store_true", help="training for mim and summary"
    )

    parser.add_argument(
        "--per_save_steps", default=1000, type=int, help="training for mim and summary"
    )
    parser.add_argument(
        "--sortish_sampler", action="store_true", help="training for mim and summary"
    )
    parser.add_argument(
        "--parallel_mode", default="", type=str, help="if use distributed ,set 'distributed' "
    )
    parser.add_argument(
        "--dataloader_num_workers", default=0, type=int, help="dataloader num_works"
    )
    args = parser.parse_args()
    # torch.backends.cudnn.deterministic = True
    # if 'OMP_NUM_THREADS' not in os.environ:
    #     os.environ["OMP_NUM_THREADS"] = str(int(multiprocessing.cpu_count() / args.nproc_per_node))

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory {args.overwrite_output_dir} already exits and is not empty. Use --overwrite_output_dir"
            f" to overcome"
        )

    # Setup CUDA GPU & distribution
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    init_logger(args.log_file)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # Setup logging
    logging.warning(
        f"Processing rank: {args.local_rank}, device:{device}, n_gpu: {args.n_gpu}, "
        f"distribution training: {bool(args.local_rank != -1)}, 16-bit training: {args.fp16}"
    )

    # Set seed
    set_seed(args)
    # 保证每次训练的结果都一样
    torch.backends.cudnn.deterministic = True

    # define model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first processing in distributed training will download model & vacab
        torch.distributed.barrier()

    config = BartConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        static_positions_embeddings=False,
        output_attention=False,
        output_hidden_states=False,
        layernorm_embedding=args.layernorm_embedding,
        add_final_layer_norm=args.add_final_layer_norm,

    )
    tokenizer = BartTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    model = EncDecQaMimBart.from_pretrained(
        args.load_checkpoint,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    embedding = model.resize_token_embeddings(50264)

    if args.local_rank == 0:
        torch.distributed.barrier()
    args.pad_token_id = tokenizer.convert_tokens_to_ids(["<pad>"])[0]
    model.to(args.device)

    logger.info("saving Config")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(config, os.path.join(args.output_dir, "bart_config.pt"))

    logger.info(f"Training/evaluate args{ args}")

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training

    if args.do_train:
        data_init = DataTrainer(args=args, tokenizer=tokenizer)
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataloader = data_init.get_train_dataloader()

        eval_dataloader = data_init.get_eval_dataloader()

        global_steps, tr_loss = trainer(
            args,
            train_dataloader,
            eval_dataloader,
            model
        )

        logger.info(f"global_steps = {global_steps}, average_loss = {tr_loss}")

    # Saving model and optimizer and tokneizer
    if args.do_train and (args.local_rank == -1 or dist.get_rank() == 0):
        # Create output directory if needed
        last_save_path = os.path.join(args.output_dir, "last_checkpoint")
        if not os.path.exists(last_save_path) and args.local_rank in [-1, 0]:
            os.makedirs(last_save_path)

        logger.info("Saving last model checkpoint to {}".format(last_save_path))

        # Saving a trained model, configuration and tokenizer using "save_pretrained()"
        # They can then be reloaded using "from_pretrained()"
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(last_save_path)
        tokenizer.save_pretrained(last_save_path)

        # Saving args
        torch.save(args, os.path.join(args.output_dir, "args.bin"))

        logger.info("Training has complete.")


if __name__ == "__main__":
    main()