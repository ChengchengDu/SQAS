
from tqdm import tqdm

import json

import random
from multiprocessing import Pool

import os
import hashlib

import gc

from nltk.tokenize import sent_tokenize

import argparse
import subprocess
import shutil



# 名词 名词复数  人称代词  名词代词
POS = ["NN", "NNS", "PRP", "NNP"]
NER_LABEL = [
    "PERSON", "ORGANIZATION", "LOCATION", "CITY", "PERCENT", "NUMBER", "MONEY", "DATE", "COUNTRY", "DATE", "TIME",
    "STATE_OR_PROVINCE", "COUNTRY", "NATIONALITY"
]


def hashhex(s):
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def write_source_to_path(args):
    """将source target分别写到文件中"""
    source_path = os.path.join(args.data_dir, args.data_type + ".source")
    target_path = os.path.join(args.data_dir, args.data_type + ".target")
    save_path = args.save_path
    source_save_path = os.path.join(args.save_path, args.data_type + '_source')
    target_save_path = os.path.join(args.save_path, args.data_type + '_target')

    if not os.path.exists(source_save_path):
        os.makedirs(source_save_path)

    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path)

    sources_new = []
    targets_new = []
    index_ = 0
    with open(source_path, 'r', encoding="utf-8") as source, \
            open(target_path, 'r', encoding="utf-8") as target:
        source_lines = source.readlines()
        target_lines = target.readlines()
        for index, (source_line, target_line) in tqdm(enumerate(zip(source_lines, target_lines))):
            if not source_line.strip() or not target_line.strip():
                continue
            if len(source_line.strip()) < len(target_line.strip()):
                continue
            # if len(source_line.strip().split()) <= 50:
            #     continue
            sources_new.append(source_line.strip() + "\n")
            targets_new.append(target_line.strip() + "\n")

            with open(os.path.join(source_save_path, str(index_) + ".txt"), 'w', encoding="utf-8") as sf:
                sf.write(source_line.strip())

            with open(os.path.join(target_save_path, str(index_) + ".txt"), 'w', encoding="utf-8") as tf:
                tf.write(target_line.strip())

            index_ += 1

    with open(os.path.join(save_path, args.data_type + ".source"), 'w', encoding="utf-8") as w:
        w.writelines(sources_new)

    with open(os.path.join(save_path, args.data_type + ".target"), 'w', encoding="utf-8") as w:
        w.writelines(targets_new)
    assert len(sources_new) == len(targets_new)
    assert len(sources_new) == index_
    print(f"after dealing, remain {index_} document")


def write_ggw_source_to_path(args):
    """将source target分别写到文件中"""
    source_path = os.path.join(args.data_dir, args.data_type + ".src.txt")
    target_path = os.path.join(args.data_dir, args.data_type + ".tgt.txt")
    save_path = args.save_path
    source_save_path = os.path.join(args.save_path, args.data_type + '_source')  # 保存的子文件的路径
    target_save_path = os.path.join(args.save_path, args.data_type + '_target')

    if not os.path.exists(source_save_path):
        os.makedirs(source_save_path)

    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path)

    sources_new = []
    targets_new = []
    index_ = 0
    with open(source_path, 'r', encoding="utf-8") as source, \
            open(target_path, 'r', encoding="utf-8") as target:
        source_lines = source.readlines()
        target_lines = target.readlines()
        for index, (source_line, target_line) in tqdm(enumerate(zip(source_lines, target_lines))):
            if not source_line.strip() or not target_line.strip():
                continue
            if len(source_line.strip()) < len(target_line.strip()):
                continue
            # if len(source_line.strip().split()) <= 50:
            #     continue
            sources_new.append(source_line.strip() + "\n")
            targets_new.append(target_line.strip() + "\n")

            with open(os.path.join(source_save_path, str(index_) + ".txt"), 'w', encoding="utf-8") as sf:
                sf.write(source_line.strip())

            with open(os.path.join(target_save_path, str(index_) + ".txt"), 'w', encoding="utf-8") as tf:
                tf.write(target_line.strip())

            index_ += 1

    with open(os.path.join(save_path, args.data_type + ".source"), 'w', encoding="utf-8") as w:
        w.writelines(sources_new)

    with open(os.path.join(save_path, args.data_type + ".target"), 'w', encoding="utf-8") as w:
        w.writelines(targets_new)
    assert len(sources_new) == len(targets_new)
    assert len(sources_new) == index_
    print(f"after dealing, remain {index_} document")


def stanford_openie(params):
    """这个是针对一个目录下的多个文件进行处理
    """

    candidate_target_dir, ie_path, mapping_dir = params.candidate_target_dir, params.ie_dir, params.mapping_dir
    candidate_dir = os.path.abspath(candidate_target_dir)
    ie_dir = os.path.abspath(ie_path)
    if not os.path.exists(ie_dir):
        os.makedirs(ie_dir)
    print("Preparing to ie %s to %s..." % (candidate_dir, ie_dir))
    candidates = os.listdir(candidate_dir)
    # make IO list file
    print("Making list of files to ie...")
    map_file = os.path.join(mapping_dir, "mapping_for_corenlp_openie.txt")
    # map_file = "/home/jazhan/code/QaExsuBart/data/mapping_for_corenlp_openie.txt"
    with open(map_file, "w") as f:
        for c in candidates:  # 对每一个文件进行处理
            f.write("%s\n" % (os.path.join(candidate_dir, c)))  # 所有处理的文本文件的路径
    command = ['java', '-Xmx100g', '-cp', '/home/jazhan/stanford-corenlp-4.2.0/*',
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
               'tokenize,ssplit,pos,lemma,depparse,ner,natlog,openie', '-threads', '40',
               '-openie.resolve_coref', 'openie.max_entailments_per_clause', '500', '-ner.useSUTime', 'false',
               '-ner.applyNumericClassifiers', 'false',
               '-filelist',
               map_file, '-outputFormat', 'json', '-outputDirectory', ie_dir]  # 关系抽取的过程
    print("IE %i files in %s and saving in %s..." % (len(candidates), candidate_dir, ie_dir))
    subprocess.call(command)
    print("Stanford CoreNLP IE has finished.")
    os.remove(map_file)

    # Check that the IE directory contains the same number of files as the original directory
    num_orig = len(os.listdir(candidate_dir))
    num_ie = len(os.listdir(ie_dir))
    if num_orig != num_ie:
        raise Exception(
            "The candidate directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                candidate_dir, num_ie, candidates, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (candidate_dir, ie_dir))

    gc.collect()


def stanford_openie_ggw(params):
    """这个是针对一个目录下的多个文件进行处理
    """
    candidate_target_dir, ie_path, mapping_dir = params.candidate_target_dir, params.ie_dir, params.mapping_dir
    candidate_dir = os.path.abspath(candidate_target_dir)
    ie_dir = os.path.abspath(ie_path)
    if not os.path.exists(ie_dir):
        os.makedirs(ie_dir)
    print("Preparing to ie %s to %s..." % (candidate_dir, ie_dir))
    candidates = os.listdir(candidate_dir)
    # make IO list file
    print("Making list of files to ie...")
    map_file = os.path.join(mapping_dir, "mapping_for_corenlp_openie.txt")
    # map_file = "/home/jazhan/code/QaExsuBart/data/mapping_for_corenlp_openie.txt"
    with open(map_file, "w") as f:
        for c in candidates:  # 对每一个文件进行处理
            f.write("%s\n" % (os.path.join(candidate_dir, c)))  # 要解析的文本文件    所有处理的文本文件的路径
    command = ['java', '-Xmx100g', '-cp', '/home/jazhan/stanford-corenlp-4.2.0/*',
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
               'tokenize,ssplit,pos,lemma,depparse,ner,natlog,openie', '-threads', '40',
               '-openie.resolve_coref', 'openie.max_entailments_per_clause', '500', '-ner.useSUTime', 'false',
               '-ner.applyNumericClassifiers', 'false',
               '-filelist',
               map_file, '-outputFormat', 'json', '-outputDirectory', ie_dir]  # 关系抽取的过程
    print("IE %i files in %s and saving in %s..." % (len(candidates), candidate_dir, ie_dir))
    subprocess.call(command)
    print("Stanford CoreNLP IE has finished.")
    os.remove(map_file)

    # Check that the IE directory contains the same number of files as the original directory
    num_orig = len(os.listdir(candidate_dir))
    num_ie = len(os.listdir(ie_dir))
    if num_orig != num_ie:
        raise Exception(
            "The candidate directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                candidate_dir, num_ie, candidates, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (candidate_dir, ie_dir))

    gc.collect()


# 有问题
def get_one_document_queries(source_line, ie_data, candidate_ie_data):

    cur_answers = []   # 当前文档符合条件的所有的问答对
    cur_tokens, cur_ner, cur_openie = get_ie(ie_data)
    # 以下构建问题和问题的答案位置
    # 1.当前目标摘要中存在关系三元组 从当前三元组中提取关系   首先考虑的是目标摘要的三元组
    if cur_openie:
        for ie in cur_openie:
            # 1) 原文中寻找subject
            subject = ie['subject']
            ie_span = ie['subject'] + " " + ie['relation'] + " " + ie['object']

            pos, context_span = check_max_context(subject, ie_span, source_line)   # 得到当前span在原文的最佳位置
            if pos != -1:
                query = get_query(subject, ie, cur_tokens, cur_ner, type="subject")
                if query:
                    cur_answers.append({"answer": subject, "pos": pos,
                                        "query": query, "ie_span": ie_span, "context_span": context_span})

            # 2) 原文中寻找object
            if not cur_answers:
                object = ie['object']
                # 得到文档出现的所有位置
                ie_span = ie['subject'] + " " + ie['relation'] + " " + ie['object']
                pos, context_span = check_max_context(object, ie_span, source_line)
                if pos != -1:
                    # 如果位置存在一定是有答案的
                    query = get_query(object, ie, cur_tokens, cur_ner, type="object")
                    if query:
                        # answer_span表示的是答案所在的上下文
                        cur_answers.append({"answer": object, "pos": pos,
                                            "query": query, "ie_span": ie_span, 'context_span': context_span})

    # 在候选的摘要中进行寻找
    if not cur_answers:
        # 需要在抽取摘要中设置问题
        extract_tokens, extract_ner, extract_openie = get_ie(candidate_ie_data)
        # 1.找openie的信息
        if extract_openie:
            for ie in extract_openie:
                # 遍历所有的ie信息 找到query
                subject = ie['subject']
                extract_ie_span = ie['subject'] + " " + ie['relation'] + " " + ie['object']
                # 如果存在关系三元组 一定使用关系三元组构建答案
                pos, context_span = check_max_context(subject, extract_ie_span, source_line)  # 得到当前span在原文的最佳位置
                if pos != -1:
                    query = get_query(subject, ie, extract_tokens, extract_ner, type="subject")
                    if query:
                        cur_answers.append({"answer": subject, "pos": pos, "query": query,
                                            "ie_span": extract_ie_span, "context_span": extract_ie_span})
                # 优先subject
                if not cur_answers:
                    object = ie['object']
                    extract_ie_span = ie['subject'] + " " + ie['relation'] + " " + ie['object']
                    pos, context_span = check_max_context(object, extract_ie_span, source_line)  # 得到当前span在原文的最佳位置
                    if pos != -1:
                        query = get_query(object, ie, extract_tokens, extract_ner, type="object")
                        if query:
                            pos = source_line.find(extract_ie_span)
                            cur_answers.append({"answer": object, "pos": pos, "query": query,
                                                "ie_span": extract_ie_span, "context_span": extract_ie_span})

        # 目标摘要和抽取到的摘要没有一个关系三元组  使用ner构建问题
        elif not cur_answers and extract_ner:
            # 使用ner得到问题
            for ner_item in cur_ner.items():

                pos = source_line.find(ner_item[0].lower())
                tag = ner_item[1]
                if tag == "PERSON":
                    new_str = "Who"
                else:
                    new_str = "What"
                if pos != -1:
                    # 寻找一个包含ner信息的上下文的span构造问题
                    context_span_start = pos - 20 if pos - 20 > 0 else 0
                    context_span_end = pos + 20 if pos + 20 < len(source_line) else len(source_line)
                    context_span = " ".join(source_line[context_span_start: context_span_end].split()[1:-1])
                    context_span = context_span.replace(source_line[pos: pos + len(ner_item[0])], new_str,
                                                        1)  # 实际上就是一种mask策略
                    query = context_span
                    cur_answers.append({"answer": ner_item[0], "pos": pos, "query": query,
                                        "ie_span": context_span, "context_span": context_span})

        elif not cur_answers and extract_tokens:
            # 使用名词得到问题
            for token_item in cur_tokens.items():
                tag = token_item[1]
                pos = source_line.find(token_item[0])
                if tag in POS and pos != -1:
                    if tag == "RPR":
                        new_str = "Who"
                    else:
                        new_str = "What"
                    context_span_start = pos - 20 if pos - 20 > 0 else 0
                    context_span_end = pos + 20 if pos + 20 < len(source_line) else len(source_line)
                    context_span = " ".join(source_line[context_span_start: context_span_end].split()[1:-1])
                    context_span = context_span.replace(source_line[pos: pos + len(token_item[0])], new_str,
                                                        1)  # 实际上就是一种mask策略
                    query = context_span
                    cur_answers.append({"answer": token_item[0], "pos": pos, "query": query,
                                        "ie_span": context_span, "context_span": context_span})

    if not cur_answers:
        # 啥都没有 强制构建问题
        source_len = len(source_line.split())
        pos = random.randint(source_len // 4, source_len // 2)
        context_span_start = pos - 3 if pos - 3 > 0 else 0
        context_span_end = pos + 3 if pos + 3 < source_len else source_len
        context_span = source_line.split()[context_span_start: context_span_end + 1]
        mask = random.randint(0, len(context_span) - 1)
        # 确定位置信息
        word = context_span[mask]
        pos = source_line.find(" ".join(context_span)) + " ".join(context_span).find(word)
        context_span[mask] = "What"
        query = " ".join(context_span)
        cur_answers.append({"answer": word, "pos": pos, "query": query,
                            "ie_span": context_span, "context_span": context_span})

    # 从当前的所有的问题随机的选择一个构建json
    answers_len = len(cur_answers)
    answer = cur_answers[random.randint(0, answers_len - 1)]
    # 继续增加data中的其他组件  实际上返回的是一个字典
    return answer


# 构建一个json的data
def _format_to_json(params):
    """从抽取到的目标摘要的三元组中构建问题 在原文中找到答案的位置  其中答案的位置需要进行检查 选择overlap最大的span的答案的位置
    作为最终的答案  需要增加一个target的分句信息
    """

    source_file, target_file, ie_file, candidate_ie_file = params  # 构成一条数据
    if not os.path.exists(ie_file) or not os.path.exists(source_file) or not os.path.exists(candidate_ie_file):
        raise ValueError("ie_json file or source file not exits")

    with open(ie_file, 'r', encoding="utf-8") as ie_f, open(source_file) as sf, \
            open(target_file) as tf, open(candidate_ie_file) as cf:
        # 获取ie信息
        ie_data = json.load(ie_f)
        candidate_ie_data = json.load(cf)
        # 获取原文档
        source_line = sf.readline().strip()
        target_line = tf.readline().strip()

        # 得到一个文档对应的问题  以及问题的答案和对应的span  返回的是一个字典
        _, _, source_line = doc_sentences_to_list(source_line, args)
        summary_start_poses, summary_end_poses, target_line = doc_sentences_to_list(target_line,args)

        candidate_answer = get_one_document_queries(source_line=source_line,
                                                    ie_data=ie_data, candidate_ie_data=candidate_ie_data)


        # 根据当前返回的字典构建json文件 data_item
        cur_data = {}
        cur_data['title'] = source_line[:25]
        paragraphs = []

        paragraph = {}
        paragraph['context'] = source_line.strip()
        paragraph['summary_text'] = target_line.strip()

        # 对summary进行分句子 然后得到每一个自然句子的开始和结束的下标
        # 用于后面互信息的计算时选择最相似的进行计算
        paragraph['summary_start_poses'] = summary_start_poses
        paragraph['summary_end_poses'] = summary_end_poses
        qas = []
        cur_qas = {}
        cur_qas['question'] = candidate_answer['query']
        cur_qas['id'] = hashhex(candidate_answer['query'])
        answers = []
        answer = {}
        answer['answer_start'] = candidate_answer['pos']
        answer['text'] = candidate_answer['answer']
        answer['context_span'] = candidate_answer['context_span']
        answer['ie_span'] = candidate_answer['ie_span']
        answers.append(answer)
        cur_qas['answers'] = answers
        qas.append(cur_qas)
        paragraph['qas'] = qas
        paragraphs.append(paragraph)
        cur_data['paragraphs'] = paragraphs

    # 继续增加data中的其他组件
    return cur_data


def doc_sentences_to_list(document, args):
    sent_start_pos = []
    sent_end_pos = []
    nature_sents = sent_tokenize(document)
    document_temp = []
    for i, sent in enumerate(nature_sents):
        if len(sent.split()) < args.min_src_ntokens_per_sent:
            continue
        document_temp.append(sent)
        cur_document = ' '.join(document_temp)
        start_pos = cur_document.rfind(sent)
        end_pos = start_pos + len(sent) - 1
        sent_start_pos.append(start_pos)
        sent_end_pos.append(end_pos)

    document = ' '.join(document_temp)

    assert len(sent_start_pos) == len(sent_end_pos), "make sure the length of start pos is equal to that of end pos"
    return sent_start_pos, sent_end_pos, document


def format_to_json(args):
    """使用多线程处理原数据集到模型指定的输入形式 读取整个数据集 处理成 json类型，"""
    if args.dataset_type != '':
        dataset_type = [args.dataset_type]
    else:
        dataset_type = ['train', 'val']  # 分别处理训练集 验证集 和 测试集

    for corpus_type in dataset_type:
        corpus_source_path = os.path.join(args.data_path, corpus_type + "_source")   # 小文件的路径  每一个原文和摘要对都写进了一个文件中
        corpus_target_path = os.path.join(args.data_path, corpus_type + "_target")
        target_ie_path = os.path.join(args.target_ie_path, corpus_type)      # 存放从target中抽取到的实体三元组的信息的文件夹
        candidate_ie_path = os.path.join(args.candidate_ie_path, corpus_type)

        # 获取目录中的每一个文件
        source_files = os.listdir(corpus_source_path)
        target_files = os.listdir(corpus_target_path)
        target_ie_files = os.listdir(target_ie_path)
        candidate_ie_files = os.listdir(candidate_ie_path)

        # 对多有的路径文件进行排序  一一对应
        source_files.sort(key=lambda x: int(x.split('.')[0]))
        target_files.sort(key=lambda x: int(x.split('.')[0]))
        target_ie_files.sort(key=lambda x: int(x.split('.')[0]))
        candidate_ie_files.sort(key=lambda x: int(x.split('.')[0]))

        source_files_list = [os.path.join(corpus_source_path, f)for f in source_files]
        target_files_list = [os.path.join(corpus_target_path, f) for f in target_files]
        target_ie_files_list = [os.path.join(target_ie_path, f) for f in target_ie_files]
        candidate_ie_files_list = [os.path.join(candidate_ie_path, f) for f in candidate_ie_files]
        assert len(source_files_list) == len(target_files_list)
        assert len(source_files_list) == len(target_ie_files_list)
        assert len(source_files_list) == len(candidate_ie_files_list)
        assert len(candidate_ie_files_list) == len(target_ie_files_list)
        dct = {'data': [], 'version': args.data_name + "-" + corpus_type + "-squad1.1"}  # 最终的json字典

        # 构建进程池
        pool = Pool(args.n_cpu)
        # 使用data接受多进程处理的对象
        data = list(tqdm(
            pool.imap(
                _format_to_json, zip(source_files_list, target_files_list, target_ie_files_list,
                                     candidate_ie_files_list), chunksize=40), total=len(source_files),
            desc="process source and target to qa json"))

        pool.close()
        pool.join()

        for data_item in data:
            if data_item:
                dct['data'].append(data_item)

        print("data has {} examples".format(len(dct['data'])))

        if not os.path.exists(args.save_json_path):
            os.makedirs(args.save_json_path)
        write_path = os.path.join(args.save_json_path, dct["version"] + ".json")

        with open(write_path, 'w', encoding="utf-8") as f:
            json.dump(dct, f, indent=4)
        print("Saving json to write_path".format(write_path))

        gc.collect()


def get_ngram(n, words):
    """
    calculate n-grams
    :param n: n-gram
    :param words:  list of tokens
    :return: a set of n-grams   set中的每一个ngram都是一个元组
    """

    ngram_set = set()
    text_length = len(words)
    # 考虑到一个token或者是2个token的情况
    if len(words) == n:
        ngram_set.add(tuple(words))
    else:

        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start):
            ngram_set.add(tuple(words[i: i + n]))

    return ngram_set


def get_word_ngrams(n, sentences):
    """
    calculate word n-grams for multiple sentences
    :param n:   n-gram
    :param sentences:  list： including some sentences   sentences中的每一个元素都是分词后的句子列表
   :return:
    """

    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])  # 所有单词的列表 包含重复的元素

    return get_ngram(n, words)


def cal_rouge(evaluate_ngrams, reference_ngrams):
    """reference 表示的是gold summary"""
    reference_count = len(reference_ngrams)
    evaluate_count = len(evaluate_ngrams)

    overlapping_ngrams = evaluate_ngrams.intersection(reference_ngrams)  # 获得交集
    overlapping_count = len(overlapping_ngrams)

    if evaluate_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluate_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    score_dict = {
        "p": precision,
        "r": recall,
        "f": f1_score
    }

    return score_dict


def indexstr(str1,str2):
    '''查找指定字符串str1包含指定子字符串str2的全部位置，
    以列表形式返回'''
    lenth2 = len(str2)
    lenth1 = len(str1)
    indexstr2 = []
    i = 0
    while str2 in str1[i:]:
        indextmp = str1.index(str2, i, lenth1)
        indexstr2.append(indextmp)
        i = (indextmp + lenth2)
    return indexstr2


def check_max_context(span, ie_span, source_line):
    """检查一个span的subject 或者是object 与关系抽取三元组最符合的原文中的位置"""

    # 得到文档出现的所有位置

    span_positions = indexstr(source_line, span)
    # if span.find(".") != -1 or span.find("*") != -1:
    #     span_positions = indexstr(source_line, span)
    # else:
    #     span_positions = [i.start() for i in re.finditer(span, source_line)]
    context_len = len(ie_span) + 15
    max_rouge = 0
    start_pos = -1
    answer_span = ""
    # 如果这个span存在的 选择span存在的最大的上下文作为答案筛选
    if span_positions:
        for pos in span_positions:
            # context_len = len(ie['relation']) + len(ie['object']) + 30
            sub_start = pos - context_len if (pos - context_len) > 0 else 0
            sub_end = pos + context_len if (pos + context_len) < len(source_line) else len(source_line)
            sub_context = source_line[sub_start: sub_end]
            sub_context = " ".join(sub_context.split()[1: -1])  # 去掉两边非token字符  得到每一个位置的上下文
            evaluate_one_gram = get_word_ngrams(1, [sub_context.split()])
            reference_one_gram = get_word_ngrams(1, [ie_span.split()])  #

            evaluate_two_gram = get_word_ngrams(2, [sub_context.split()])
            reference_two_gram = get_word_ngrams(2, [ie_span.split()])

            rouge1 = cal_rouge(evaluate_one_gram, reference_one_gram)['f']
            rouge2 = cal_rouge(evaluate_two_gram, reference_two_gram)['f']

            rouge = (rouge1 + rouge2) / 2

            if rouge > max_rouge:
                start_pos = pos
                max_rouge = rouge
                answer_span = sub_context

    return start_pos, answer_span


def get_query(span, ie, cur_tokens, cur_ner, type="subject"):
    """根据抽取到的span三元组的实体类型或者是名词类型得到问题的形式
       不是三元组的情况额外处理
    """
    subject = ie['subject']
    relation = ie["relation"]
    object = ie["object"]

    query = ""
    if type == "subject":
        if span in cur_ner:  # 在实体中
            if cur_ner[span] == "PERSON":
                query = "Who " + relation + " " + object + " ?"
            elif cur_ner[span] == "DATE" or cur_ner[span] == "TIME":
                query = "When " + relation + " " + object + " ?"
            else:  # 其他的实体
                query = "Which " + cur_ner[span].lower() + " " + relation + " " + object + " ?"
        elif span in cur_tokens or any([v in POS for v in cur_tokens.values()]):  # 不在实体列表中  在名词中
            query = "What " + relation + " " + object + " ?"
        else:
            query = "What " + relation + " " + object + " ?"

    elif type == "object":
        if span in cur_ner:
            if cur_ner[span] == "PERSON":
                query = "Who " + "does " + subject + " " + relation + " ?"
            elif cur_ner[span] == "DATE" or cur_ner[span] == "TIME":
                query = "When " + "does " + subject + " " + relation + " ?"
            elif cur_ner[span] == "PERCENT" or cur_ner[span] == "NUMBER":
                query = "How many " + "does " + subject + " " + relation + " ?"
            elif cur_ner[span] == "MONEY":
                query = "How much " + "does " + subject + " " + relation + " ?"
            else:
                query = "Where " + "does " + subject + " " + relation + " ?"
        elif span in cur_tokens or any([w in cur_tokens for w in span.split()]):
            query = "What does " + subject + " " + relation + " ?"
        else:
            query = "What does " + subject + " " + relation + " " + "?"
    else:
        # 不是三元组中部分
        pass

    return query


def get_ie(data):
    """一条data就是一个输入的信息 得到一条数据中**所有**的 ie中的token 实体 关系等信息"""

    # 得到当前摘要中的所有的token 实体 关系抽取等信息
    cur_tokens = {}
    cur_ner = {}
    cur_openie = []  # 按照subject抽取关系  没有重复   对于相同的subject选择长度最长的关系三元组

    sentences = data["sentences"]  # 包含7项的字典
    for sentence in sentences:

        # 获取当前句子的单词和词性
        if sentence["tokens"]:
            tokens = sentence["tokens"]
            for tok in tokens:
                cur_tokens[tok["word"]] = tok["pos"]

        # 获取当前句子的ner信息
        if sentence["entitymentions"]:
            entities = sentence["entitymentions"]
            for entity in entities:
                cur_ner[entity['text']] = entity['ner']

        if sentence["openie"]:
            openies = sentence["openie"]
            final_ie = None
            max_ie_len = 0
            last_subject = None
            # 同名的找最长的三元组
            for openie in openies:
                subject = openie["subject"]
                relation = openie["relation"]
                object = openie["object"]
                if last_subject is None:
                    last_subject = subject
                    max_ie_len = len(subject) + len(relation) + len(object)
                    final_ie = openie
                else:
                    if subject != last_subject:
                        cur_openie.append(final_ie)  ## 将同名subject最大的长度关系三元组抽取
                        last_subject = None
                        max_ie_len = 0
                        final_ie = None
                    else:
                        cur_ie_len = len(subject) + len(relation) + len(object)
                        if cur_ie_len > max_ie_len:
                            max_ie_len = cur_ie_len
                            final_ie = openie
                            last_subject = subject
            if final_ie is not None:
                cur_openie.append(final_ie)
    if not cur_tokens:
        print("data", data)

    return cur_tokens, cur_ner, cur_openie


def write_lead3_to_path(save_dir, source_dir):
    """将抽取到的摘要写到各自的文件夹中 用于测试集中"""

    test_path = os.path.join(save_dir, "test_candidate_summary")
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    source_path = os.path.join(source_dir, 'test.source')

    candidate_summaries = []    # 所有的候选摘要
    with open(source_path, 'r', encoding="utf-8") as source:
        source_lines = source.readlines()
        for index, source_line in tqdm(enumerate(source_lines)):
            cur_summary = []
            # 处理每一个句子到自然句,并且去除掉短句子
            source_sents = sent_tokenize(source_line.strip())

            if len(source_sents) < 3:
                length = len(source_sents)
            else:
                length = 3

            for i in range(length):
                cur_summary.append(source_sents[i].strip())  # 将前三句话加进去  作为当前的抽取式摘要

            summary = " ".join(cur_summary)
            candidate_summaries.append(summary + "\n")
            with open(os.path.join(test_path, str(index)+".txt"), 'w', encoding="utf-8") as wf:
                wf.write(summary)

        assert len(source_lines) == len(candidate_summaries)
    with open(os.path.join(save_dir, "candidate.txt"), "w", encoding="utf-8") as f:
        f.writelines(candidate_summaries)

    print("after dealing test.source, its length is {}".format(len(candidate_summaries)))

# def doc_sentences_to_list(document):
#     """将目标摘要进行分句 得到每一个自然句子的开始位置和结束位置"""
#     sent_start_pos = []
#     sent_end_pos = []
#     nature_sents = sent_tokenize(document)
#     document_temp = []
#     for i, sent in enumerate(nature_sents):
#         document_temp.append(sent)
#         cur_document = ' '.join(document_temp)
#         start_pos = cur_document.rfind(sent)
#         end_pos = start_pos + len(sent) - 1
#         sent_start_pos.append(start_pos)
#         sent_end_pos.append(end_pos)
#
#     assert len(sent_start_pos) == len(sent_end_pos), "make sure the length of start pos is equal to that of end pos"
#     return sent_start_pos, sent_end_pos


def get_test_query(ie_path, candidate_path, save_path):
    """ie_path中是包含多个文件的"""
    if not os.path.exists(ie_path) or not os.path.exists(candidate_path):
        raise ValueError("ie_path should be exits")
    ie_files = os.listdir(ie_path)
    ie_files = ie_files.sort(key=lambda x: int(x.split()[0]))
    candidate_files = os.listdir(candidate_path)
    candidate_files = candidate_files.sort(key=lambda x: int(x.split()[0]))
    assert len(candidate_files) == len(ie_files)
    all_queries = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    answer_path = os.path.join(save_path, 'answers')
    if not os.path.exists(answer_path):
        os.makedirs(answer_path)
    for index, (ie_file, candidate_file) in enumerate(zip(ie_files, candidate_files)):
        with open(ie_file, 'r', encoding="utf-8") as f:
            ie_data = json.load(f)
        with open(candidate_file, 'r', encoding="utf=8") as cf:
            candidate_line = cf.readline().strip()
        cur_answers = []  # 当前文档符合条件的所有的问答对
        cur_tokens, cur_ner, cur_openie = get_ie(ie_data)
        # 以下构建问题和问题的答案位置
        # 1.当前目标摘要中存在关系三元组 从当前三元组中提取关系   首先考虑的是目标摘要的三元组
        if cur_openie:
            for ie in cur_openie:
                # 1) 原文中寻找subject
                subject = ie['subject']
                ie_span = ie['subject'] + " " + ie['relation'] + " " + ie['object']

                query = get_query(subject, ie, cur_tokens, cur_ner, type="subject")
                if query:
                    cur_answers.append({"answer": subject,
                                        "query": query, "ie_span": ie_span})

                # 2) 原文中寻找object
                if not cur_answers:
                    object = ie['object']
                    # 得到文档出现的所有位置
                    ie_span = ie['subject'] + " " + ie['relation'] + " " + ie['object']
                    query = get_query(object, ie, cur_tokens, cur_ner, type="object")
                    if query:
                        # answer_span表示的是答案所在的上下文
                        cur_answers.append({"answer": object,
                                            "query": query, "ie_span": ie_span})
        if not cur_answers and cur_ner:
            # 按照ner构建问题
            # 使用ner得到问题
            for ner_item in cur_ner.items():
                pos = candidate_line.find(ner_item[0].lower())
                tag = ner_item[1]
                if tag == "PERSON":
                    new_str = "Who"
                else:
                    new_str = "What"
                if pos != -1:
                    # 寻找一个包含ner信息的上下文的span构造问题
                    context_span_start = pos - 20 if pos - 20 > 0 else 0
                    context_span_end = pos + 20 if pos + 20 < len(candidate_line) else len(candidate_line)
                    context_span = candidate_line[context_span_start: context_span_end].replace(ner_item[0], new_str)
                    context_span = " ".join(context_span.split()[1:-1])
                    query = context_span
                    cur_answers.append({"answer": ner_item[0], "query": query,
                                        "ie_span": context_span})

        elif not cur_answers and cur_tokens:
            # 使用名词得到问题
            for token_item in cur_tokens.items():
                tag = token_item[1]
                pos = candidate_line.find(token_item[0])
                if tag in POS and pos != -1:
                    if tag == "RPR":
                        new_str = "Who"
                    else:
                        new_str = "What"
                    context_span_start = pos - 20 if pos - 20 > 0 else 0
                    context_span_end = pos + 20 if pos + 20 < len(candidate_line) else len(candidate_line)
                    context_span = candidate_line[context_span_start: context_span_end].replace(ner_item[0], new_str)
                    context_span = " ".join(context_span.split()[1:-1])
                    query = context_span
                    cur_answers.append({"answer": token_item[0],  "query": query,
                                        "ie_span": context_span, })
        else:
            candidate_len = len(candidate_line.split())
            pos = random.randint(candidate_len // 4, candidate_len // 2)
            context_span_start = pos - 3 if pos - 3 > 0 else 0
            context_span_end = pos + 3 if pos + 3 < candidate_len else candidate_len
            context_span = candidate_line.split()[context_span_start: context_span_end + 1]
            mask = random.randint(0, len(context_span) - 1)
            # 确定位置信息
            word = context_span[mask]
            pos = candidate_line.find(" ".join(context_span)) + " ".join(context_span).find(word)
            context_span[mask] = "What"
            query = " ".join(context_span)
            cur_answers.append({"answer": word, "query": query,
                                "ie_span": context_span})

        # 随机选择一个问题
        answers_len = len(cur_answers)
        answer = cur_answers[random.randint(0, answers_len - 1)]
        all_queries.append(answer['query'].strip() + "\n")
        with open(os.path.join(answer_path + str(index)+".json"), 'w', encoding="utf-8") as qf:
            json.dumps(answer, qf)
    with open(os.path.join(save_path, "all_query.txt"), 'w', encoding="utf-8") as f:
        f.writelines(all_queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # write_source_path args
    parser.add_argument(
        '--data_dir', default="./", type=str
    )
    parser.add_argument(
        '--save_path', default="./", type=str
    )
    parser.add_argument(
        '--data_type', default="train", type=str
    )

    # stanford_openie
    parser.add_argument(
        '--candidate_target_dir', default="./", type=str
    )
    parser.add_argument(
        '--ie_dir', default="./", type=str
    )
    parser.add_argument(
        '--mapping_dir', default="./", type=str
    )

    # format_to_json
    parser.add_argument(
        '--dataset_type', default="train", help="train val test"
    )
    parser.add_argument(
        '--data_path', default="./", help="train_source dir path "
    )
    parser.add_argument(
        '--target_ie_path', default="./", help="xsum or cnndm target ie dir"
    )
    parser.add_argument(
        '--candidate_ie_path', default="./", help="xsum cnndm candidate target ie dir"
    )
    parser.add_argument(
        '--data_name', default="xsum", help="data name for deal"
    )
    parser.add_argument(
        '--n_cpu', default=40, type=int
    )
    parser.add_argument(
        '--save_json_path', default="./", help="saving json path ", type=str,
    )
    parser.add_argument(
        '--min_src_ntokens_per_sent', default=3, type=int
    )

    # write_lead3_to_path
    parser.add_argument(
        '--save_dir', type=str, default="./"
    )
    parser.add_argument(
        '--source_dir', type=str, default="./"
    )
    # get_test_query
    parser.add_argument(
        '--test_ie_path', type=str, default="./"
    )
    parser.add_argument(
        '--test_candidate_path', type=str, default="./"
    )
    parser.add_argument(
        '--test_save_path', type=str, default="./"
    )

    args = parser.parse_args()

    # write_source_to_path(args)

    # write_ggw_source_to_path(args)

    # stanford_openie(args)

    # stanford_openie_ggw(args)

    format_to_json(args)

    # write_lead3_to_path(args.save_dir, args.source_dir)

    # get_test_query(args.test_ie_path, args.test_candidate_path, args.test_save_path)

