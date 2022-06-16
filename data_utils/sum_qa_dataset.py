import os
import logging
import json
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import collections
import numpy as np

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.tokenization_bart import BartTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""将得到的摘要问答json文件处理成features cache, 以便dataset的加载"""

"""args.sum  args.squad, args.sum_doc_stride, args.extract_answer_num=3"""

BOS = "<s>"
CLS = "<s>"
PAD = "<pad>"
SEP = "</s>"
UNK = "<unk>"
MASK = "<mask>"


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def get_sum_squad_examples_and_features(
        tokenizer, file, args, output_examples=False, is_training=True, is_test=False):
    """将json文件处理成examples和features"""
    logger.info("Creating example and features from data file at %s", args.data_dir)
    examples = read_sum_squad_examples(file, is_training=is_training, debug=args.debug)

    features = sum_squad_convert_examples_to_features(
                        examples,
                        tokenizer,
                        args.max_seq_length,
                        args.sum_doc_stride,
                        args.max_query_length,
                        is_training
             )
    print("features len", len(features))
    # cached-train-cnn-512
    if is_training:
        name = "train"
    elif is_test:
        name = "test"
    else:
        name = "val"

    cached_features_file = os.path.join(
        args.data_save,
        "cached-{}-{}-{}-{}".format(
            name,
            args.sum_data_type,
            "features",
            str(args.max_seq_length),
        )
    )

    logger.info("Saving features into cached file to  %s", cached_features_file)
    torch.save(features, cached_features_file)

    if output_examples:
        cached_examples_file = os.path.join(
            args.data_save,
            "cached-{}-{}-{}-{}".format(
                name,
                args.sum_data_type,
                "examples",
                str(args.max_seq_length)
            )
        )

        logger.info(f"Creating cached_example_file -{cached_examples_file}")
        logger.info("Saving example into cached file to  %s", cached_examples_file)
        torch.save(examples, cached_examples_file)


def read_sum_squad_examples(
        input_file, is_training, version_2_with_negative=False,
        debug=False):
    """Read a Summary-SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    # 去除空格和末尾换行
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    if debug:
        input_data = input_data[:5]

    for entry in input_data:
        paragraphs = entry["paragraphs"]
        for paragraph in paragraphs:
            paragraph_text = paragraph["context"]
            summary_text = paragraph["summary_text"]

            # 同样需要对summary的token进行分词  按照空格进行分词
            sum_tokens = []
            sum_char_to_word_offset = []
            prev_is_whitespace = True
            for c in summary_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        sum_tokens.append(c)
                    else:
                        sum_tokens[-1] += c
                    prev_is_whitespace = False
                sum_char_to_word_offset.append(len(sum_tokens) - 1)   # 将每个字符对应的token数进行对应

            summary_start_poses = paragraph['summary_start_poses']
            summary_end_poses = paragraph['summary_end_poses']
            sum_start_positions, sum_end_positions = [], []    # 词级别的位置信息
            for sum_start_pos, sum_end_pos in zip(summary_start_poses, summary_end_poses):
                # print("sum_start_pos", sum_start_pos)
                # print("summary_text", summary_text)
                sum_start_positions.append(sum_char_to_word_offset[sum_start_pos])
                sum_end_positions.append(sum_char_to_word_offset[sum_end_pos])
                assert len(sum_start_positions) == len(sum_end_positions)



            doc_tokens = []   # 按照空格分词得到的doc_tokens
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)   # 将每个字符对应的token数进行对应

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]

                # orig_answer_text = None   # 如果是验证集的话 orig_answer_text是None 不能用于评估
                is_impossible = False
                start_position = None
                end_position = None
                orig_answer_text = None   # 每一个抽取到的候选摘要的列表集合
                answers = None
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if not is_impossible:
                        answer = qa["answers"][0]       # 在summary qa中answer最多包含三个句子 并且通过最长的单词数进行限制
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset +
                                                           answer_length - 1]

                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            continue

                    else:
                        start_position = -1   # 如果是impossible的话会初始化start_pos 和 end_pos为-1
                        end_position = -1
                        orig_answer_text = ""

                else:
                    # 其实在摘要的问答中式不需要进行对原文的评估的  只是对ground truth计算rouge即可
                    answers = qa['answers']     # 增加answers的部分用于评估

                example = SumSquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    sum_start_positions=sum_start_positions,
                    sum_end_positions=sum_end_positions,
                    summary_tokens=sum_tokens,
                    is_impossible=is_impossible,
                    answers=answers)
                examples.append(example)

    return examples


class SumSquadExample(object):
    """
       A single training/test example for the Summary Squad dataset.
       For examples without an answer, the start and end position are -1.
       """
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 sum_start_positions=None,
                 sum_end_positions=None,
                 summary_tokens=None,
                 is_impossible=None,
                 answers=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.summary_tokens = summary_tokens
        self.is_impossible = is_impossible
        self.answers = answers
        self.sum_start_positions = sum_start_positions
        self.sum_end_positions = sum_end_positions


def sum_squad_convert_example_to_features(
        example, max_seq_length,
        doc_stride, max_query_length, is_training
        ):
    """Loads a data file into a list of `InputBatch`s.单进程处理一个example"""

    features = []

    query_tokens = tokenizer.tokenize(" " + example.question_text.rstrip())
    # summary_tokens = [BOS] + tokenizer.tokenize(" " + example.summary_text.rstrip()) + [SEP]
    # summary_tokens 需要按照doc的做法处理 不能直接tokenize

    sum_tok_to_orig_index = []  # tok表示BPE之后的tok
    sum_orig_to_tok_index = []
    all_sum_tokens = []
    for (i, token) in enumerate(example.summary_tokens):
        sum_orig_to_tok_index.append(len(all_sum_tokens))   # 只是针对文档进行处理的 没有加入query
        sub_tokens = tokenizer.tokenize(" " + token)          # 需要对每一个token的前面加一个空格Bart的分词格式
        for sub_token in sub_tokens:
            sum_tok_to_orig_index.append(i)
            all_sum_tokens.append(sub_token)


    # all_sum_tokens = [BOS] + all_sum_tokens + [SEP]
    # 对摘要的自然句子的分界进行处理
    sum_tok_start_positions = []
    sum_tok_end_positions = []
    for sum_token_start_pos, sum_token_end_pos in zip(example.sum_start_positions, example.sum_end_positions):
        if sum_token_end_pos < len(example.summary_tokens)  - 1:
            sum_token_end_pos = sum_orig_to_tok_index[sum_token_end_pos + 1] - 1
        else:
            sum_token_end_pos = len(all_sum_tokens) - 1
        sum_tok_start_positions.append(sum_orig_to_tok_index[sum_token_start_pos])
        sum_tok_end_positions.append(sum_token_end_pos)
    assert len(sum_tok_start_positions) == len(sum_tok_end_positions)
    all_sum_tokens = [BOS] + all_sum_tokens + [SEP]
    for i, pos in enumerate(sum_tok_start_positions):
        sum_tok_start_positions[i] += 1
        sum_tok_end_positions[i] += 1

    query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
    # if len(query_tokens) > max_query_length-2:    # 注意query同样需要加特殊词  注意如果增加实体之后的问题的长度
    #     query_tokens = query_tokens[0:max_query_length-2]

    # 补全摘要的长度到最大长度
    if len(all_sum_tokens) < max_seq_length:
        diff = max_seq_length - len(all_sum_tokens)
        summary_tokens = all_sum_tokens + [PAD] * diff
    else:
        summary_tokens = all_sum_tokens[: max_seq_length]

    summary_ids = tokenizer.convert_tokens_to_ids(summary_tokens)

    tok_to_orig_index = []   # tok表示BPE之后的tok
    orig_to_tok_index = []
    all_doc_tokens = []

    # 获取BPE之后的tok和未分词之前的token位置对应关系
    for (i, token) in enumerate(example.doc_tokens):

        orig_to_tok_index.append(len(all_doc_tokens))   # 只是针对文档进行处理的 没有加入query
        sub_tokens = tokenizer.tokenize(" " + token)          # 需要对每一个token的前面加一个空格Bart的分词格式
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)  # 目前得到的是基于doc的开始和结束的位置

    # The -4 accounts for bart： [CLS] query [SEP] [CLS] doc [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 4

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0    # 实际上是每一个span在原文中的开始位置
    # 将原文按照doc_stride确定span的开始位置 span的长度由max_tokens_for_doc决定的

    while start_offset < len(all_doc_tokens):    # 对文档进行span的划分
        length = len(all_doc_tokens) - start_offset   # 文档中剩余的单词数
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))  # 以max_tokens_for_doc长度作为一个span进行划分
        if len(doc_spans) == 1:
            # print("choose first max_tokens_for_doc{} for summary.".format(max_tokens_for_doc))
            break
        if start_offset + length == len(all_doc_tokens):   # 对文档中所有的token都进行了划分
            break
        start_offset += min(length, doc_stride)

    # 对所有的doc_span 进行遍历 确定模型的输入
    # 如果只选择一个 那么只进行一次遍历操作
    for (doc_span_index, doc_span) in enumerate(doc_spans):    # 在每一个span中进行input_ids的构建
        tokens = []              # 模型输入的token_ids
        token_to_orig_map = {}   # 当前span的token对的位置相对于原文未分词之前的tok的位置映射
        token_is_max_context = {}   # 判断当前的tok是不是处于最大的上下文
        #  注意bart中没有segment_ids
        tokens.append(BOS)

        for token in query_tokens:
            tokens.append(token)
        tokens.append(SEP)

        tokens.append(BOS)     # 文章主体部分同样采用 CLS + doc + SEP的形式
        # 检查当前span下的句子的分界是否仍然存在

        # 对当前span的context token进行统计
        context_tokens = list()
        context_tokens.append(BOS)    # 区分context tokens和input tokens  context tokens表示的是文章  tokens表示的是加入query之后得到的模型的输入
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i   # doc_span.start表示tok在BPE分词之后的doc中的索引下标
            token_to_orig_map[len(
                tokens)] = tok_to_orig_index[split_token_index]    # 在span中加入query之后的tok的在原始文档中（未BPE分词）中的索引下标
            # 检验split_token_index是不是在上下文最大的窗口的span中
            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)    # doc_span_index表示的划分的第几个span， split_token_index表示的是分词之后的tokend的index
            token_is_max_context[len(tokens)] = is_max_context   # 表示当前的tok是不是处在最大的上下文的span中
            tokens.append(all_doc_tokens[split_token_index])
            context_tokens.append(all_doc_tokens[split_token_index])
        tokens.append(SEP)
        context_tokens.append(SEP)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(tokens)

        assert len(tokens) == len(input_mask)

        # Zero-pad up to the sequence length.
        while len(tokens) < max_seq_length:
            tokens.append(PAD)
            input_mask.append(0)
        if len(tokens) > max_seq_length:
            tokens = tokens[: max_seq_length]
            input_mask = input_mask[: max_seq_length]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids = np.asarray(input_ids, dtype=np.int32)
        input_mask = np.asarray(input_mask, dtype=np.uint8)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        start_position = None    # BPE之后答案在span中加入问题之后的位置
        end_position = None

        if is_training and not example.is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start     # 表示当前span在原始文档中开始的位置
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False

            # 针对summary-squad数据来说 规定如果答案中的三个句子一个都不包含那么就去除
            # 对于其中只有一两个句子属于答案的 需要重新确定答案的位置 答案的个数
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True  # 说明答案的开始和结束的位置不是在当前的span中

            if out_of_span:
                start_position = 0
                end_position = 0

            else:   # 当前的span包含答案
                doc_offset = len(query_tokens) + 3
                start_position = tok_start_position - doc_start + doc_offset  # 加入了query的偏移量
                end_position = tok_end_position - doc_start + doc_offset
            if out_of_span:
                continue

        if is_training and example.is_impossible:
            start_position = 0
            end_position = 0

        features.append(
            SumInputFeatures(
                unique_id=0,
                example_index=0,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                answer_texts=example.orig_answer_text,
                q_ids=query_ids,
                summary_ids=summary_ids,
                start_position=start_position,
                end_position=end_position,
                sum_start_positions=sum_tok_start_positions,
                sum_end_positions=sum_tok_end_positions,
                is_impossible=example.is_impossible))   # start_position表示的是当前span下答案的开始和结束的位置（包含问题）
    return features


def sum_squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    threads=30,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features in order to deal datasets.
    多进程处理代码

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=sum_squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            sum_squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    return features


def sum_squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


class SumInputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 answer_texts,
                 input_mask,
                 q_ids,
                 summary_ids=None,
                 start_position=None,
                 end_position=None,
                 sum_start_positions=None,
                 sum_end_positions=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.answer_texts = answer_texts
        self.input_mask = input_mask
        self.q_ids = q_ids
        self.summary_ids = summary_ids
        self.start_position = start_position
        self.end_position = end_position

        self.sum_start_positions = sum_start_positions
        self.sum_end_positions = sum_end_positions
        self.is_impossible = is_impossible


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.检查当前的span是不是不是当前position位置token的最大上下文span"""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same因为窗口大小是相同的, of course).     maximum的上下文选择方式是选left token较少的
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def check_sentence_boundary(cur_span, sentence_start_positions, sentence_end_positions, doc_offset):
    """检查当前的span中包含的完整句子的界限"""
    sentence_start_positions_ = []  # 移除掉句子span之外的句子的边界
    sentence_end_positions_ = []

    doc_start = cur_span.start
    doc_end = cur_span.start + cur_span.length - 1   # 这个开始和结束的位置是针对BPE之后的tok的 相对于document的位置信息 没有包含答案
    # 检索每一个位置是不是越界
    for index, sentence_start_pos in enumerate(sentence_start_positions):
        sentence_end_pos = sentence_end_positions[index]
        if sentence_start_pos >= doc_start and sentence_end_pos <= doc_end:
            cur_sentence_start_pos = sentence_start_pos - doc_start + doc_offset
            cur_sentence_end_pos = sentence_end_pos - doc_start + doc_offset
            sentence_start_positions_.append(cur_sentence_start_pos)
            sentence_end_positions_.append(cur_sentence_end_pos)
    assert len(sentence_start_positions_) == len(sentence_end_positions_)

    return sentence_start_positions_, sentence_end_positions_


def get_segment_ids(doc_offset, sentence_start_positions, sentence_end_positions, doc_span):
    segment_ids = []

    if sentence_start_positions[0] == doc_offset:
        cur_segment_id = 1

    else:
        cur_segment_id = 1
        segment_ids.extend([cur_segment_id] * (sentence_start_positions[0] - doc_offset))
        cur_segment_id = 1 - cur_segment_id

    segment_start_pos = sentence_start_positions[0]
    for end_pos in sentence_end_positions:
        segment_ids.extend([cur_segment_id] * (end_pos - segment_start_pos + 1))
        segment_start_pos = end_pos + 1
        cur_segment_id = 1 - cur_segment_id

    # 可能结束的位置之后还有token 因此需要补充segment ids
    last_segment_ids = (doc_span.length - len(segment_ids)) * [cur_segment_id]
    segment_ids = segment_ids + last_segment_ids

    assert len(segment_ids) == doc_span.length, f"len(segment_ids)-{len(segment_ids)} != doc_len-{doc_span.length}"
    return segment_ids


class SummarySquadDataset(Dataset):
    def __init__(self, args, file_name="cached-train-cnn-features-512"):
        features_dir = args.sum_features_data_dir
        features_path = os.path.join(features_dir, file_name)
        logger.info(f"Loading features from cached dir -{features_path}")
        self.pad_token_id = args.pad_token_id
        self.features = torch.load(features_path)
        self.labels = ["B-A", "I-A", "E-A", "O"]  # 预测每一个token的序列 是一个四分类的问题

    def __getitem__(self, item):
        feature = self.features[item]
        input_ids = feature.input_ids
        input_mask = feature.input_mask
        segment_ids = feature.segment_ids
        start_positions = feature.start_positions
        end_positions = feature.end_positions
        sentence_start_positions = feature.sentence_start_positions
        sentence_end_positions = feature.sentence_end_positions

        # 对当前序列中的所有的token打标签
        label_dict = {}  # 记录答案中的每一个token的标签
        tag_labels = []
        # 对一条数据进行处理
        for start_pos, end_pos in zip(start_positions, end_positions):
            length = end_pos - start_pos + 1
            for r in range(length):
                if r == 0:
                    label_dict[start_pos] = 0
                if r == length - 1:
                    label_dict[start_pos + r] = 2
                else:
                    label_dict[start_pos + r] = 1
        for i in range(len(input_ids)):
            if i in label_dict:
                tag_labels.append(label_dict[i])
            else:
                tag_labels.append(3)
        assert len(input_ids) == len(tag_labels), \
            f"len_input_ids:{len(input_ids)} == len_tag_labels:{len(tag_labels)}"

        input_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "segment_ids": segment_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "sentence_start_positions": sentence_start_positions,
            "sentence_end_positions": sentence_end_positions,
            "tag_labels": tag_labels
        }
        return input_dict

    def __len__(self):
        return len(self.features)

    def collate_fn(self, batch):
        """将batch中的list进行stack 得到batch tensor形式"""
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        segment_ids = torch.stack([torch.tensor(x["segment_ids"]) for x in batch])

        # start_positions是抽取到的句子的位置
        start_positions = torch.stack([torch.tensor(x["start_positions"]) for x in batch])
        end_positions = torch.stack([torch.tensor(x["end_positions"]) for x in batch])
        tag_labels = torch.stack([torch.tensor(x["tag_labels"]) for x in batch])
        max_sentences_len = max([len(x["sentence_start_positions"]) for x in batch])
        for x in batch:
            sentence_start_positions = x["sentence_start_positions"]
            sentence_end_positions = x["sentence_end_positions"]
            assert len(sentence_start_positions) == len(sentence_end_positions)
            if len(sentence_start_positions) < max_sentences_len:
                diff = max_sentences_len - len(sentence_start_positions)
                sentence_start_positions.extend([-1] * diff)
                sentence_end_positions.extend([-1] * diff)
            x["sentence_start_positions"] = sentence_start_positions
            x["sentence_end_positions"] = sentence_end_positions

        sentence_start_positions = torch.stack([torch.tensor(x["sentence_start_positions"]) for x in batch])
        sentence_end_positions = torch.stack([torch.tensor(x["sentence_end_positions"]) for x in batch])
        if tag_labels is not None:
            input_ids, attention_mask, segment_ids, start_positions, end_positions, tag_labels = self.trim_batch(
                input_ids,
                attention_mask,
                segment_ids,
                start_positions,
                end_positions,
                tag_labels,
            )
        else:
            input_ids, attention_mask, segment_ids, start_positions, end_positions, tag_labels = self.trim_batch(
                input_ids,
                attention_mask,
                segment_ids,
                start_positions,
                end_positions,
            )

        batch_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "segment_ids": segment_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "sentence_start_positions": sentence_start_positions,
            "sentence_end_positions": sentence_end_positions,
            "tag_labels": tag_labels
        }

        return batch_inputs   # 返回一个训练batch

    def trim_batch(self, input_ids, attention_mask, segment_ids, start_positions, end_positions, tag_labels=None):
        """去除掉input_ids的batch中全为0的列"""
        keep_column_mask = input_ids.ne(self.pad_token_id).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        segment_ids = segment_ids[:, keep_column_mask]
        start_positions = start_positions[:,  keep_column_mask]
        end_positions = end_positions[:,  keep_column_mask]
        if tag_labels is not None:
            tag_labels = tag_labels[:, keep_column_mask]
            return input_ids, attention_mask, segment_ids, start_positions, end_positions, tag_labels
        return input_ids, attention_mask, segment_ids, start_positions, end_positions


def get_SummarySquad_dataloader(args, file_name, type_data):

    dataset = SummarySquadDataset(args, file_name=file_name)
    if type_data == "train":
        args.train_batch_size = args.per_gpu_train_batch_size_summary_squad * max(1, args.n_gpu)
        sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler,
            batch_size=args.train_batch_size,
            collate_fn=dataset.collate_fn
        )
    else:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.eval_batch_size_squad,
            collate_fn=dataset.collate_fn
        )
    return dataloader


def read_source_and_target(source_path, target_path):
    source_lines = open(source_path, 'r', encoding="utf-8").readlines()
    target_lines = open(target_path, 'r', encoding="utf-8").readlines()
    item = 0
    for index, (source_line, target_line) in enumerate(zip(source_lines, target_lines)):
        if len(source_line) < len(target_line):
            item += 1
            print("wrong")
            print(index)
    print(item)

