#!/bin/bash
# 通过json文件处理cnndm xsum为features和examples
python get_summary_features.py \
--model_name_or_path facebook/bart-large \
--data_dir /home/jazhan/code/query_based_summarization/data_utils/dealed_data/ggw/ggw-val-squad1.1.json \
--sum_data_type ggw \
--max_seq_length 256 \
--max_query_length 20 \
--cache_example \
--sum_doc_stride 256 \
--is_training \
--data_save /home/jazhan/code/query_based_summarization/data_utils/dealed_data/ggw/valid/
