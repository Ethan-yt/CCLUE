export TASK_NAME=$1
export MODEL_NAME=$2

if [[ TASK_NAME == 'quote' ]]; then
  ARGS=$(
    echo '--crf'
    echo '--crf_lr 5e-3'
  )
else
  ARGS=''
fi

python3 run_sequence_labeling.py $ARGS \
  --model_name_or_path $MODEL_NAME \
  --dataset_name data/punctuation \
  --task_name $TASK_NAME \
  --output_dir output/$MODEL_NAME/$TASK_NAME \
  --evaluation_strategy steps \
  --logging_steps 100 \
  --fp16 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --do_train \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --overwrite_output_dir \
  --warmup_ratio 0.1 \
  --do_predict
