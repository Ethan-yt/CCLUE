export TASK_NAME=fspc_poem
export MODEL_NAME=$1

python3 run_classification.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name data/$TASK_NAME \
  --task_name $TASK_NAME \
  --output_dir output/$MODEL_NAME/$TASK_NAME \
  --evaluation_strategy steps \
  --logging_steps 20 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 1024 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --load_best_model_at_end \
  --metric_for_best_model accuracy \
  --do_predict \
  --do_train \
  --overwrite_output_dir \
  --overwrite_cache \
  --warmup_ratio 0.1 \
  --max_seq_length 40


