import argparse
import importlib
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader
from torchvision import datasets,transforms
import json
import os

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of dataset;', type=str, default='mnist')
    parser.add_argument('--num_clients', help='the number of clients;', type=int, default=100)
    parser.add_argument('--dist', help='type of distribution;', type=int, default=0)
    parser.add_argument('--skew', help='the degree of niid;', type=float, default=0.5)
    parser.add_argument("--idx_path", type=str)
    parser.add_argument("--task_name",type=str)
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, option =dict()):
        super(TaskGen, self).__init__(benchmark='cifar10',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/cifar10/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json
        self.data_path = option['idx_path']
    
    def run(self):
        """ Generate federated task"""
        # check if the task exists
        if not self._check_task_exist():
            self.create_task_directories()
        else:
            
            print("Task Already Exists.")
            return
        # read raw_data into self.train_data and self.test_data
        print('-----------------------------------------------------')
        print('Lading...')
        self.load_data()
        print('Done.')
        # partition data and hold-out for each local dataset
        print('-----------------------------------------------------')
        print('Partitioning data...')
        with open(self.data_path, "r") as f:
            loaded_data_idx = json.load(f)
        
        local_datas = []
        for key in loaded_data_idx.keys():
            local_datas.append(list(loaded_data_idx[key]))

        # breakpoint()
        # local_datas = self.partition()
        train_cidxs, valid_cidxs = self.local_holdout(local_datas, rate=0.8, shuffle=True)
        print('Done.')
        # save task infomation as .json file and the federated dataset
        print('-----------------------------------------------------')
        print('Saving data...')
        self.save_info()
        self.save_data(train_cidxs, valid_cidxs)
        print('Done.')
        return
    
    def load_data(self):
        self.train_data = datasets.CIFAR10(self.rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        self.test_data = datasets.CIFAR10(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    def convert_data_for_saving(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x':train_x, 'y':train_y}
        self.test_data = {'x': test_x, 'y': test_y}

if __name__ == '__main__':
    option = read_option()
    generator = TaskGen(dist_id = option['dist'], skewness = option['skew'], num_clients=option['num_clients'], option = option)
    generator.run()
