import logging
import os
import random
import math
import torch.distributed as dist
import numpy as np
import torch
from typing import Iterable
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler, RandomSampler

from transformers.file_utils import cached_property
from typing import List
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# train and valid的dataset  evaluate的时候需要重新写
# dataloader这一块应该是没问题的
class SummarySquadDataset(Dataset):
    def __init__(self, args, tokenizer, file_name="cached-train-cnn-features-512", data_type="train"):
        features_dir = args.sum_features_data_dir
        features_path = os.path.join(features_dir, file_name)
        logger.info(f"Loading features from cached dir -{features_path}")
        self.pad_token_id = args.pad_token_id
        self.features = torch.load(features_path)
        if args.debug:
            self.features = self.features[:1]
            print(self.features)
        self.src_lens = self.get_inputs_lens(self.features, self.pad_token_id)
        # if data_type == "valid":
        #     self.features = self.features   # 这样做会不会让每次测试的结果不一致？
        self.tokenizer = tokenizer

    @staticmethod
    def get_inputs_lens(features, pad_token_id):
        """得到每一个输入的src的除pad之外的input_ids的长度 是按照BPE之后token的个数进行统计的
        原fairseq是按照字符进行统计的  按照实际的样本的长度进行排序"""
        return [feature.input_ids.__ne__(pad_token_id).sum() for feature in features]

    # 需要重新修改判断是不是分布式的
    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def __getitem__(self, item):
        feature = self.features[item]
        input_ids = feature.input_ids
        input_mask = feature.input_mask
        target_ids = feature.summary_ids
        start_position = feature.start_position
        end_position = feature.end_position
        # 增加summary的自然句子的分界处理
        sum_start_positions = feature.sum_start_positions
        sum_end_positions = feature.sum_end_positions

        input_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "target_ids": target_ids,
            "start_position": start_position,
            "end_position": end_position,
            "sum_start_positions": sum_start_positions,
            "sum_end_positions": sum_end_positions

        }
        return input_dict

    def __len__(self):
        return len(self.features)

    def collate_fn(self, batch):
        """将batch中的list进行stack 得到batch tensor形式"""
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])

        # 这个的start_positions是以真实的位置信息作为最终的答案 这里面只有一个答案  用-1补全 计算损失的时候忽略不计
        start_positions = torch.tensor([x['start_position'] for x in batch])
        end_positions = torch.tensor([x['end_position'] for x in batch])

        # 对句子的长度进行填充  后面选择句子的时候注意将-1去掉

        target_ids = torch.stack([torch.tensor(x["target_ids"]) for x in batch])
        input_ids, attention_mask = self.trim_batch(
            input_ids,
            attention_mask,
        )
        target_ids = self.trim_target_batch(target_ids)

        # 注意这里问答的损失不是按照BCE的方式 而是squad
        batch_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }

        # 处理摘要的句子边界
        max_sents_len_sum = max([len(x['sum_start_positions']) for x in batch])
        batch_size = len(batch)
        sum_start_positions = torch.randn(batch_size, max_sents_len_sum).fill_(1)
        sum_end_positions = torch.randn(batch_size, max_sents_len_sum).fill_(1)
        for index, b in enumerate(batch):
            cur_sum_start_positions = b['sum_start_positions']
            cur_sum_end_positions = b['sum_end_positions']
            start_len = len(cur_sum_start_positions)
            end_len = len(cur_sum_end_positions)
            assert start_len == end_len
            sum_start_positions.data[index][: start_len] = torch.tensor(cur_sum_start_positions).data
            sum_end_positions.data[index][:end_len] = torch.tensor(cur_sum_end_positions).data

        batch_inputs['sum_start_positions'] = sum_start_positions
        batch_inputs['sum_end_positions'] = sum_end_positions
        return batch_inputs   # 返回一个训练batch

    def trim_batch(self, input_ids, attention_mask):
        """去除掉input_ids的batch中全为0的列"""
        keep_column_mask = input_ids.ne(self.pad_token_id).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]

        return input_ids, attention_mask

    def trim_target_batch(self, target_ids):
        keep_column_mask = target_ids.ne(self.pad_token_id).any(dim=0)
        target_ids = target_ids[:, keep_column_mask]
        return target_ids


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    def __init__(self, data, batch_size, shuffle=True):
        # data指的是每一个example的长度列表
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def key(self, i):
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i: i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            print("train distributed sampler", rank)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank: self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def read_feaures():
    features = torch.load("/home/jazhan/code/QaExsuBart/data/transformers_dealed_data/cnndm_dealed/cached-train-cnndm-features-1024")
    for feature in features:
        print(torch.tensor(feature.end_positions).sum())
        print(torch.tensor(feature.start_positions).sum())
        print("8"*10)


def read_examples():
    examples = torch.load("/home/jazhan/code/QaExsuBart/data/transformers_dealed_data/cnndm_dealed/cached-test-cnndm-examples-1024")
    for example in examples:
        print(example.start_positions)
        print(example.end_positions)


class DataTrainer(object):
    def __init__(self, args, tokenizer):
        super(DataTrainer, self).__init__()
        if args.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training process the dataset,
            # and the others will use the cache
            torch.distributed.barrier()
        self.train_dataset = SummarySquadDataset(
            args, tokenizer, file_name=args.sum_train_features_file_name)
        print("加载训练数据.....")
        if args.local_rank == 0:
            # Make sure only the first process in distributed training process the dataset,
            # and the others will use the cache
            torch.distributed.barrier()

        self.eval_dataset = SummarySquadDataset(
            args, tokenizer=tokenizer, file_name=args.sum_valid_features_file_name
        )
        print("加载验证集数据")
        self.args = args

    def get_train_sampler(self):
        if self.args.sortish_sampler:
            self.train_dataset.make_sortish_sampler(
                self.args.train_batch_size,
                distributed=(self.args.parallel_mode == "distributed"),
            )

        return (
            RandomSampler(self.train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(self.train_dataset)
        )

    def get_train_dataloader(self) -> DataLoader:

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self.get_train_sampler()
        # train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.train_dataset.collate_fn,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
        )

    def get_eval_sampler(self, eval_dataset):
        if self.eval_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset=None):

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self.get_eval_sampler(eval_dataset)
        # eval_sampler = DistributedSortishSampler(eval_dataset, self.args.eval_batch_size)
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=eval_dataset.collate_fn,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
        )


    def get_test_dataloader(self, test_dataset):
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """

        test_sampler = self.get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=test_dataset.collate,
            drop_last=False,
        )


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indices sequentially, making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training), which means that the model params won't
    have to be synced (i.e. will not hang for synchronization even if varied number of forward passes), we still add
    extra samples to the sampler to make it evenly divisible (like in `DistributedSampler`) to make it easy to `gather`
    or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"Indices length {len(indices)} and total size {self.total_size} mismatched"

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert (
            len(indices) == self.num_samples
        ), f"Indices length {len(indices)} and sample number {self.num_samples} mismatched"

        return iter(indices)

    def __len__(self):
        return self.num_samples





# if __name__ == "__main__":
#     # read_examples()
#     read_feaures()