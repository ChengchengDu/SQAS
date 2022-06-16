#! /bin/bash
# 训练ner-based query cnndm的命令
python  train.py \
--total_num_update 20000 \
--epochs -1 \
--sum_features_data_dir /home/jazhan/code/query_based_summarization/data_utils/dealed_data/ggw/  \
--output_dir /home/jazhan/code/query_based_summarization/qfs_checkpoint/ggw_checkpoint/valid/  \
--load_checkpoint facebook/bart-large \
--sum_train_features_file_name cached-val-ggw-features-256  \
--sum_valid_features_file_name cached-val-ggw-features-256 \
--do_train  \
--overwrite_output_dir \
--per_gpu_train_batch_size 4 \
--eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--logging_steps 10 \
--save_steps 10000 \
--log_dir /home/jazhan/code/query_based_summarization/qfs_checkpoint/ggw_checkpoint/valid/ggw_valid_runs \
--log_file /home/jazhan/code/query_based_summarization/qfs_checkpoint/ggw_checkpoint/ggw_valid_train.log \
--learning_rate 3e-5 \
--warmup_steps 500 \
--weight_decay 0.01 \
--sortish_sampler \
