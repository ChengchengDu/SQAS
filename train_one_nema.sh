#! /bin/bash
# 训练ner-based query nema的命令
python  train.py \
--total_num_update -1 \
--epochs 3 \
--sum_features_data_dir /home/jazhan/code/query_based_summarization/data_utils/dealed_data/nema/  \
--output_dir /home/jazhan/code/query_based_summarization/nema3_checkpoint/  \
--load_checkpoint facebook/bart-large-cnn \
--sum_train_features_file_name cached-train-nema-features-256  \
--sum_valid_features_file_name cached-val-nema-features-256  \
--do_train  \
--overwrite_output_dir \
--per_gpu_train_batch_size 2 \
--eval_batch_size 2 \
--gradient_accumulation_steps 2 \
--logging_steps 10 \
--save_steps 2500 \
--log_dir /home/jazhan/code/query_based_summarization/nema3_checkpoint/nema_runs \
--log_file /home/jazhan/code/query_based_summarization/nema3_checkpoint/nema_train.log \
--learning_rate 3e-5 \
--warmup_steps 500 \
--weight_decay 0.01 \
--sortish_sampler \
