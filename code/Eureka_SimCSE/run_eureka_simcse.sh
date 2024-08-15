#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3

#multi-lingual
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/google-bert/bert-base-multilingual-uncased \
    --dataset_name /Eureka/eureka_sup_dataset \
    --output_dir result/eureka_sup_bert-base-multilingual-uncased \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

#zh
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/google-bert/bert-base-chinese \
    --dataset_name /Eureka/eureka_zh_sup_dataset \
    --output_dir result/eureka_sup_bert-base-chinese \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

#med-zh
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/trueto/medbert-base-wwm-chinese \
    --dataset_name /Eureka/eureka_zh_sup_dataset \
    --output_dir result/eureka_sup_medbert-base-wwm-chinese \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

#en
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/google-bert/bert-base-uncased \
    --dataset_name /Eureka/eureka_en_sup_dataset \
    --output_dir result/eureka_sup_bert-base-uncased \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

#med-en
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/dmis-lab/biobert-base-cased-v1.2 \
    --dataset_name /Eureka/eureka_en_sup_dataset \
    --output_dir result/eureka_sup_biobert-base-cased-v1.2 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"


##unsup
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/google-bert/bert-base-multilingual-uncased \
    --dataset_name /Eureka/eureka_unsup_dataset \
    --output_dir result/eureka_unsup_bert-base-multilingual-uncased \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"


#zh
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/google-bert/bert-base-chinese \
    --dataset_name /Eureka/eureka_zh_unsup_dataset \
    --output_dir result/eureka_unsup_bert-base-chinese \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

#med-zh
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/trueto/medbert-base-wwm-chinese \
    --dataset_name /Eureka/eureka_zh_unsup_dataset \
    --output_dir result/eureka_unsup_medbert-base-wwm-chinese \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

#en
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/google-bert/bert-base-uncased \
    --dataset_name /Eureka/eureka_zh_unsup_dataset \
    --output_dir result/eureka_unsup_bert-base-uncased \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

#med-en
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path /pretrained_model/dmis-lab/biobert-base-cased-v1.2 \
    --dataset_name /Eureka/eureka_en_unsup_dataset \
    --output_dir result/eureka_unsup_biobert-base-cased-v1.2 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 512 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 125 \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --pooler_type cls \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --temp 0.05 \
    --seed 42 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"