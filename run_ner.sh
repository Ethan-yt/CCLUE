export TASK_NAME=ner
export MODEL_NAME=$1

if [[ $2 == 'crf' ]]; then
  ARGS=$(
    echo "--output_dir output/$MODEL_NAME/$TASK_NAME-crf"
    echo "--learning_rate 3e-5"
    echo '--crf'
    echo '--crf_lr 5e-3'
    echo "--overwrite_output_dir"
  )
else
  ARGS=$(
    echo "--output_dir output/$MODEL_NAME/$TASK_NAME"
    echo "--learning_rate 5e-5"
    echo "--overwrite_output_dir"
  )
fi

echo $CRF

python3 run_sequence_labeling.py $ARGS \
  --model_name_or_path $MODEL_NAME \
  --dataset_name data/ner \
  --task_name $TASK_NAME \
  --evaluation_strategy steps \
  --logging_steps 30 \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 24 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --fp16 \
  --do_train \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --warmup_ratio 0.1 \
  --do_predict
