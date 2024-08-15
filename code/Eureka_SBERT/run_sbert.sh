 export CUDA_VISIBLE_DEVICES=0,1,2,3

#multi-lingual
torchrun --nproc_per_node=4 --master_port=10087 train.py \
    --model_name_or_path /google-bert/bert-base-multilingual-uncased \
    --dataset_name_or_path /Eureka/eureka_sup_dataset \
    --output_dir result/eureka_sbert_bert-base-multilingual-uncased \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --do_train \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --fp16 \
    "$@"

#zh
torchrun --nproc_per_node=4 --master_port=10087 train.py \
    --model_name_or_path /google-bert/bert-base-chinese \
    --dataset_name_or_path /Eureka/eureka_zh_sup_dataset \
    --output_dir result/eureka_sbert_bert-base-chinese \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --do_train \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --fp16 \
    "$@"

# med-zh
torchrun --nproc_per_node=4 --master_port=10087 train.py \
    --model_name_or_path /trueto/medbert-base-wwm-chinese \
    --dataset_name_or_path /Eureka/eureka_zh_sup_dataset \
    --output_dir result/eureka_sbert_medbert-base-wwm-chinese \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --do_train \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --fp16 \
    "$@"

# en
torchrun --nproc_per_node=4 --master_port=10087 train.py \
    --model_name_or_path /google-bert/bert-base-uncased \
    --dataset_name_or_path /Eureka/eureka_en_sup_dataset \
    --output_dir result/eureka_sbert_bert-base-uncased \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --do_train \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --fp16 \
    "$@"

#med-en
torchrun --nproc_per_node=4 --master_port=10087 train.py \
    --model_name_or_path /dmis-lab/biobert-base-cased-v1.2 \
    --dataset_name_or_path /Eureka/eureka_en_sup_dataset \
    --output_dir result/eureka_sbert_biobert-base-cased-v1.2 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model HR@10 \
    --load_best_model_at_end \
    --do_train \
    --save_total_limit 2 \
    --overwrite_output_dir \
    --fp16 \
    "$@"