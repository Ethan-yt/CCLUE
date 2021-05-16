export MODEL_NAME=$1

python3 run_mrc.py \
  --model_name_or_path $MODEL_NAME \
  --output_dir output/$MODEL_NAME/mrc2 \
  --evaluation_strategy steps \
  --logging_steps 100 \
  --fp16 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --warmup_ratio 0.1 \
  --learning_rate 1e-5 \
  --num_train_epochs 15 \
  --do_train \
  --do_eval \
  --load_best_model_at_end \
  --metric_for_best_model accuracy \
  --max_seq_length 512 \
  --do_predict
