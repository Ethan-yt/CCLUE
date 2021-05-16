export TASK_NAME=text_classification
export MODEL_NAME=$1

python3 run_classification.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name data/$TASK_NAME \
  --task_name $TASK_NAME \
  --output_dir output/$MODEL_NAME/$TASK_NAME \
  --evaluation_strategy steps \
  --logging_steps 1000 \
  --fp16 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --load_best_model_at_end \
  --metric_for_best_model accuracy \
  --do_train \
  --overwrite_output_dir \
  --do_predict

