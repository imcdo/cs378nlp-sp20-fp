run: normal eval

normal:
	python3 main.py \
		--use_gpu \
		--model "baseline" \
		--model_path "squad_model.pt" \
		--train_path "datasets/squad_train.jsonl.gz" \
		--dev_path "datasets/squad_dev.jsonl.gz" \
		--output_path "squad_predictions.txt" \
		--hidden_dim 256 \
		--bidirectional \
		--do_train \
		--do_test \
		--vocab_size 1000000
 
train_only:
	python3 main.py \
		--use_gpu \
		--model "baseline" \
		--model_path "squad_model.pt" \
		--train_path "datasets/squad_train.jsonl.gz" \
		--dev_path "datasets/squad_dev.jsonl.gz" \
		--output_path "squad_predictions.txt" \
		--hidden_dim 256 \
		--bidirectional \
		--do_train 

visualize:
	python3 visualize.py --path datasets/squad_train.jsonl.gz --samples 1

eval:
	python3 evaluate.py \
    --dataset_path "datasets/squad_dev.jsonl.gz" \
    --output_path "squad_predictions.txt"

bert:
	python3 main.py \
		--use_gpu \
		--model "baseline" \
		--model_path "squad_model.pt" \
		--train_path "datasets/squad_train.jsonl.gz" \
		--dev_path "datasets/squad_dev.jsonl.gz" \
		--output_path "squad_predictions.txt" \
		--hidden_dim 256 \
		--bidirectional \
		--do_train \
		--do_test \
		--rnn_cell_type bert\
		--vocab_size 1000000 

		run:
load:
	python3 main.py \
		--use_gpu \
		--model "baseline" \
		--model_path "squad_model.pt" \
		--train_path "datasets/squad_train.jsonl.gz" \
		--dev_path "datasets/squad_dev.jsonl.gz" \
		--output_path "squad_predictions.txt" \
		--hidden_dim 256 \
		--bidirectional \
		--do_train \
		--do_test \
		--vocab_size 1000000 \
		--load_pickle samples
 write:
	python3 main.py \
		--use_gpu \
		--model "baseline" \
		--model_path "squad_model.pt" \
		--train_path "datasets/squad_train.jsonl.gz" \
		--dev_path "datasets/squad_dev.jsonl.gz" \
		--output_path "squad_predictions.txt" \
		--hidden_dim 256 \
		--bidirectional \
		--do_train \
		--do_test \
		--vocab_size 1000000 \
		--write_pickle samples

write-%:
	python3 main.py \
		--use_gpu \
		--model "baseline" \
		--model_path "squad_model.pt" \
		--train_path "datasets/squad_train.jsonl.gz" \
		--dev_path "datasets/squad_dev.jsonl.gz" \
		--output_path "squad_predictions.txt" \
		--hidden_dim 256 \
		--bidirectional \
		--do_train \
		--do_test \
		--vocab_size 1000000 \
		--write_pickle $*

load-%:
	python3 main.py \
		--use_gpu \
		--model "baseline" \
		--model_path "squad_model.pt" \
		--train_path "datasets/squad_train.jsonl.gz" \
		--dev_path "datasets/squad_dev.jsonl.gz" \
		--output_path "squad_predictions.txt" \
		--hidden_dim 256 \
		--bidirectional \
		--do_train \
		--do_test \
		--vocab_size 1000000 \
		--load_pickle $*