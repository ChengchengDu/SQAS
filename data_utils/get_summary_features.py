"""获取将摘要数据集处理成问答形式的features"""

import argparse
import sys
sys.path.append("/home/jazhan/code/query_based_summarization/")
from data_utils.sum_qa_dataset import get_sum_squad_examples_and_features
# from data.sum_qa_dataset_1 import get_sum_squad_examples_and_features

from transformers.tokenization_bart import BartTokenizer


def main():
    parser = argparse.ArgumentParser()
 
    parser.add_argument(
        '--model_name_or_path', default="facebook/bart-large", type=str, help="pretrained model for tokenizer"
    )
    parser.add_argument(
        '--data_dir', default=None, type=str, help="summarySquad.json path"
    )
    parser.add_argument(
        '--sum_data_type', default="cnn", type=str, help="choice for cnn, dm , xsum, nyt"
    )
    parser.add_argument(
        '--max_seq_length', default=512, type=int, help="max length of input sequence"
    )

    parser.add_argument(
        '--debug', action="store_true", help="whether is debug"
    )
    parser.add_argument(
        '--max_query_length', default=20, type=int, help="max length of query sequence"
    )
    parser.add_argument(
        '--cache_example', action="store_true", help="whether cached example"
    )
    parser.add_argument(
        '--sum_doc_stride', default=256, type=int, help="stride window size for input_ids"
    )
    parser.add_argument(
        '--is_training', action="store_true", help="is training?"
    )
    parser.add_argument(
        '--data_save', default=None, help="features or example saved path"
    )
    parser.add_argument(
        '--is_test', action="store_true", help="is test dataset"
    )

    args = parser.parse_args()

    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

    get_sum_squad_examples_and_features(
        tokenizer=tokenizer,
        file=args.data_dir,
        args=args,
        output_examples=args.cache_example,
        is_training=args.is_training,
        is_test=args.is_test
    )


if __name__ == "__main__":
    main()
