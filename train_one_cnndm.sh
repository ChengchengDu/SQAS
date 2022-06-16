#! /bin/bash
# 训练ner-based query cnndm的命令
python  train.py \
--total_num_update 25000 \
--epochs -1 \
--sum_features_data_dir /home/jazhan/code/query_based_summarization/data_utils/dealed_data/cnndm/  \
--output_dir /home/jazhan/code/query_based_summarization/cnndm_checkpoint/  \
--load_checkpoint facebook/bart-large \
--sum_train_features_file_name cached-train-cnndm-features-1024  \
--sum_valid_features_file_name cached-val-cnndm-features-1024  \
--do_train  \
--overwrite_output_dir \
--per_gpu_train_batch_size 2 \
--eval_batch_size 2 \
--gradient_accumulation_steps 16 \
--logging_steps 10 \
--save_steps 5000 \
--log_dir /home/jazhan/code/query_based_summarization/cnndm_checkpoint/cnndm_runs \
--log_file /home/jazhan/code/query_based_summarization/cnndm_checkpoint/cnndm_train.log \
--learning_rate 3e-5 \
--warmup_steps 500 \
--weight_decay 0.01 \
--sortish_sampler \
