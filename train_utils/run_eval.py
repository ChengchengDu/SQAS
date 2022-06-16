# 最新transformer官方的评估代码
# 使用自己处理的数据squad进行评估 加入segment部分
import argparse
from ast import arg
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append("/home/jcdu/code/QaEnDeBart/")
import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

# from data.test_data import SummaryTestDataset
sys.path.append('/home/jazhan/code/qfs_pure/train_utils/')
from test_data import SummaryTestDataset
from transformers.tokenization_bart import BartTokenizer
from transformers.configuration_bart import BartConfig
from transformers.modeling_bart import BartForConditionalGeneration
from rouge_score import rouge_scorer, scoring
from typing import List, Dict

logger = getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]

def calculate_rouge(output_lns: List[str], reference_lns: List[str], use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}


# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i : i + n]

# 仔细检查
# 看一下是不是可以添加segment -ids
def generate_summaries_or_translations(
    args,
    out_file: str,
    model_path: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    decoder_start_token_id=None,
    top_k=None,
    **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    fout = Path(out_file).open("w", encoding="utf-8")
    config = BartConfig.from_pretrained(
        args.model_name,
        static_positions_embeddings=False,
        output_attention=False,
        output_hidden_states=False,
    )
    print("config", config)
    model_path = str(model_path)
    model = BartForConditionalGeneration(config=config).from_pretrained(
        pretrained_model_name_or_path=model_path).to(
        device)
    if fp16:
        model = model.half()
    if config.vocab_size == 50264:
        embedding = model.resize_token_embeddings(50265) 

    tokenizer = BartTokenizer.from_pretrained(args.model_name)

    args.pad_token_id = tokenizer.convert_tokens_to_ids(["<pad>"])[0]

    # 评估的数据是要处理成query document summary三个txt文件的
    test_dataset = SummaryTestDataset(
        tokenizer=tokenizer, data_dir=args.data_dir, type_path=args.data_type, shuffle_query=args.shuffle_query)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate_fn,
        drop_last=False,
        shuffle=False
    )
    test_iterator = tqdm(test_dataloader, desc="Test Iterator")
    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(test_iterator):
            model.eval()
            source_ids = batch["source_ids"].long().to(args.device)
            attention_mask = batch["source_mask"].long().to(args.device)

            summaries = model.generate(
                input_ids=source_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=decoder_start_token_id,
                max_length=args.max_length,
                min_length=args.min_length,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                num_return_sequences=args.num_return_sequences,
                top_k=args.top_k,
                **generate_kwargs,
            )
            dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for hypothesis in dec:
                fout.write(hypothesis + "\n")
                fout.flush()

        fout.close()
        runtime = int(time.time() - start_time)  # seconds
        n_obs = len(test_dataset)
        return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))


# class SummarySquadDataset(Dataset):
#     def __init__(self, args, tokenizer):
#
#         features_dir = args.sum_features_data_dir
#         features_path = os.path.join(features_dir, file_name)
#         logger.info(f"Loading features from cached dir -{features_path}")
#         self.pad_token_id = args.pad_token_id
#         self.features = torch.load(features_path)
#         self.src_lens = self.get_inputs_lens(self.features, self.pad_token_id)
#         # if data_type == "valid":
#         #     self.features = self.features   # 这样做会不会让每次测试的结果不一致？
#         self.tokenizer = tokenizer
#
#     @staticmethod
#     def get_inputs_lens(features, pad_token_id):
#         """得到每一个输入的src的除pad之外的input_ids的长度 是按照BPE之后token的个数进行统计的
#         原fairseq是按照字符进行统计的  按照实际的样本的长度进行排序"""
#         return [feature.input_ids.__ne__(pad_token_id).sum() for feature in features]
#
#     # 需要重新修改判断是不是分布式的
#     def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
#         if distributed:
#             return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
#         else:
#             return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)
#
#     def __getitem__(self, item):
#         feature = self.features[item]
#         input_ids = feature.input_ids
#         input_mask = feature.input_mask
#         target_ids = feature.summary_ids
#         segment_ids = feature.segment_ids
#         start_target = feature.start_targets
#         end_target = feature.end_targets
#         start_positions = feature.start_positions
#         end_positions = feature.end_positions
#         sentence_start_positions = feature.sentence_start_positions
#         sentence_end_positions = feature.sentence_end_positions
#
#         input_dict = {
#             "input_ids": input_ids,
#             "attention_mask": input_mask,
#             "segment_ids": segment_ids,
#             "target_ids": target_ids,
#             "start_target": start_target,
#             "end_target": end_target,
#             "start_positions": start_positions,
#             "end_positions": end_positions,
#             "sentence_start_positions": sentence_start_positions,
#             "sentence_end_positions": sentence_end_positions
#         }
#         return input_dict
#
#     def __len__(self):
#         return len(self.features)
#
#     def collate_fn(self, batch):
#         """将batch中的list进行stack 得到batch tensor形式"""
#         input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
#         attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
#         segment_ids = torch.stack([torch.tensor(x["segment_ids"]) for x in batch])
#
#         # # 这个的start_targets是0101形式的
#         # start_targets = torch.stack([torch.tensor(x["start_targets"]) for x in batch])
#         # end_targets = torch.stack([torch.tensor(x["end_targets"]) for x in batch])
#
#         # 这个的start_positions是以真实的位置信息作为最终的答案  已经补全到长度是3  用-1补全 计算损失的时候忽略不计
#         start_positions = torch.stack([torch.tensor(x["start_positions"]) for x in batch])
#         end_positions = torch.stack([torch.tensor(x["end_positions"]) for x in batch])
#
#         # 对句子的长度进行填充  后面选择句子的时候注意将-1去掉
#         max_sentences_size = max([len(x["sentence_start_positions"]) for x in batch])
#         batch_size = len(batch)
#         sentence_start_positions = torch.ones(batch_size, max_sentences_size).fill_(-1)
#         sentence_end_positions = torch.ones(batch_size, max_sentences_size).fill_(-1)
#         for i in range(batch_size):
#             size = len(batch[i]["sentence_start_positions"])
#             sentence_start_positions.data[i][:size] = torch.tensor(batch[i]["sentence_start_positions"]).data
#             sentence_end_positions.data[i][:size] = torch.tensor(batch[i]["sentence_end_positions"]).data
#
#         target_ids = torch.stack([torch.tensor(x["target_ids"]) for x in batch])
#         input_ids, attention_mask, segment_ids = self.trim_batch(
#             input_ids,
#             attention_mask,
#             segment_ids
#         )
#         target_ids = self.trim_target_batch(target_ids)
#
#         # 注意这里问答的损失不是按照BCE的方式 而是squad
#         batch_inputs = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "target_ids": target_ids,
#             "segment_ids": segment_ids,
#             "start_positions": start_positions,
#             "end_positions": end_positions,
#             "sentence_start_positions": sentence_start_positions,
#             "sentence_end_positions": sentence_end_positions
#         }
#
#         return batch_inputs   # 返回一个训练batch
#
#     def trim_batch(self, input_ids, attention_mask, segment_ids):
#         """去除掉input_ids的batch中全为0的列"""
#         keep_column_mask = input_ids.ne(self.pad_token_id).any(dim=0)
#         input_ids = input_ids[:, keep_column_mask]
#         attention_mask = attention_mask[:, keep_column_mask]
#         segment_ids = segment_ids[:, keep_column_mask]
#         # start_positions = start_positions[:,  keep_column_mask]
#         # end_positions = end_positions[:,  keep_column_mask]
#
#         return input_ids, attention_mask, segment_ids
#
#     def trim_target_batch(self, target_ids):
#         keep_column_mask = target_ids.ne(self.pad_token_id).any(dim=0)
#         target_ids = target_ids[:, keep_column_mask]
#         return target_ids

def com_score(output_lines, reference_lines):
    score_fn = calculate_rouge
    output_lns = [x.rstrip() for x in output_lines]
    reference_lns = [x.rstrip() for x in reference_lines][: len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    return scores


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--model_path", type=str, help="has trained model for loaded")
    parser.add_argument("--layernorm_embedding", action="store_true", help="layernorm embedding")
    parser.add_argument(
        "--add_final_layer_norm", action="store_true"
    )
    parser.add_argument("--data_dir", type=str, help="source and query path")
    parser.add_argument("--save_path", type=str, help="where to save summaries")

    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument(
        "--score_path",
        type=str,
        required=False,
        default="metrics.json",
        help="where to save the rouge score in json format",
    )
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        '--num_beams', type=int, default=4, help="beam search"
    )
    parser.add_argument(
        '--length_penalty', type=float, default=2.0,
    )
    parser.add_argument(
        '--max_length', default=140, type=int, help="max_length for generated summary"
    )
    parser.add_argument(
        '--min_length', default=55, type=int, help="min_length for generated summary"
    )
    parser.add_argument(
        '--no_repeat_ngram_size', type=int, default=3
    )
    parser.add_argument(
        '--cache_dir', type=str, default="./"
    )
    parser.add_argument(
        "--decoder_start_token_id",
        type=int,
        default=None,
        required=False,
        help="Defaults to using config",
    )
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--early_stopping", action="store_true", help="early_stopping"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=8, required=False
    )
    parser.add_argument(
        "--shuffle_query", action="store_true"
    )
    parser.add_argument(
        "--num_return_sequences", help="the num of generated summary ", default=1, type=int)
    parser.add_argument(
        "--top_k", help="top sample num", default=5, type=int)
    parser.add_argument(
        "--top_p", help="top_p sample  ", default=0.95, type=float)
    parser.add_argument(
        '--data_type', type=str, default="test",
    )
    
    # parser.add_argument(
    #     "--sum_features_data_dir", type=str, default="./", help="feature path"
    # )
    args = parser.parse_args()

    # examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]
    # if args.n_obs > 0:
    #     examples = examples[: args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")
    runtime_metrics = generate_summaries_or_translations(
        args,
        args.save_path,
        args.model_path,
        batch_size=args.eval_batch_size,
        device=args.device,
        fp16=args.fp16,
        decoder_start_token_id=args.decoder_start_token_id,
    )
    if args.reference_path is None:
        return
    # Compute scores
    score_fn = calculate_rouge
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][: len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    scores.update(runtime_metrics)
    print(scores)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w"))
    return scores


if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate()
