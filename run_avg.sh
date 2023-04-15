python main.py --task "cifar10_cnum100_pareto" --model cnn --algorithm fedprob --num_rounds 500 --num_epochs 5 --learning_rate 0.01 --proportion 0.1 --batch_size 10 --eval_interval 1 --log_wandb  --sigma_certify 0.05 --session_name "sigma0.05"
python main.py --task "cifar10_cnum100_pareto" --model cnn --algorithm fedprob --num_rounds 500 --num_epochs 5 --learning_rate 0.01 --proportion 0.1 --batch_size 10 --eval_interval 1 --log_wandb --sigma_certify 0.1 --session_name "sigma0.1"
python main.py --task "cifar10_cnum100_pareto" --model cnn --algorithm fedprob --num_rounds 500 --num_epochs 5 --learning_rate 0.01 --proportion 0.1 --batch_size 10 --eval_interval 1 --log_wandb