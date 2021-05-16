set -e

# for model_name in "hfl/chinese-roberta-wwm-ext" "ethanyt/guwenbert-base"; do

#for lid in 6 7 8 9 10; do
#  python ./examples/retrieval/gluecc.py --normalize --data_dir ./data/retrieval --output_dir ./local-test --exp_name roberta-cos-mean${lid} --max_seq_length 512 --exp_results_dir ./local-res --mean_layer_id ${lid} --model_name "hfl/chinese-roberta-wwm-ext"
#  python ./examples/retrieval/gluecc.py --normalize --data_dir ./data/retrieval --output_dir ./local-test --exp_name roberta-cos-mean${lid}-rev --max_seq_length 512 --exp_results_dir ./local-res --mean_layer_id ${lid} --model_name "hfl/chinese-roberta-wwm-ext" --reverse_eval
#  python ./examples/retrieval/gluecc.py --normalize --data_dir ./data/retrieval --output_dir ./local-test --exp_name guwenbert-cos-mean${lid} --max_seq_length 512 --exp_results_dir ./local-res --mean_layer_id ${lid} --model_name "ethanyt/guwenbert-base"
#  python ./examples/retrieval/gluecc.py --normalize --data_dir ./data/retrieval --output_dir ./local-test --exp_name guwenbert-cos-mean${lid}-rev --max_seq_length 512 --exp_results_dir ./local-res --mean_layer_id ${lid} --model_name "ethanyt/guwenbert-base" --reverse_eval
#done
lid=8
python ./run_retrieval.py --normalize --data_dir ./data/retrieval --output_dir ./retr-test --exp_name guwenbert-fs-cos-mean${lid}-rev --max_seq_length 512 --exp_results_dir ./retr-res --mean_layer_id ${lid} --model_name 'models/guwenbert-base-fs' --reverse_eval
