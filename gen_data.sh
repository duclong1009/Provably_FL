# python generate_fedtask.py --dataset cifar10 --dist 0 --skew 0 --num_clients 100
python generate_fedtask_basedon_index.py --dataset cifar10 --dist 4 --skew 0.5 --num_clients 100
python generate_fedtask.py --dataset cifar10 --dist 2 --skew 0.8 --num_clients 100
python generate_fedtask.py --dataset cifar10 --dist 3 --skew 0.8 --num_clients 100