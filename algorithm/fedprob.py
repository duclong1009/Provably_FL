import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from .fedbase import BasicClient, BasicServer
from .fedprob_utils.smooth import Smooth
from .fedprob_utils.accuracy import ApproximateAccuracy
from main import logger
import utils.fflow as flw

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)
        self.sigma = option['sigma_certify']
        self.N0 = option['n0']
        self.N = option['n']
        self.alpha = option['alpha_certify']
        self.num_classes = option['num_classes']

    def train(self, model: nn.Module):
        """
        Training process for smoothed classifier
        Client training with noisy data
        """
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        # Traing phase for base classifer
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                inputs, outputs = batch_data
                inputs = inputs.to(self.calculator.device) + torch.rand_like(inputs, device=self.calculator.device) * self.sigma

                noisy_batch = [inputs, outputs]

                loss = self.calculator.get_loss(model, noisy_batch)
                loss.backward()
                optimizer.step()

    # TODO: change hard fix options
    def certify(self, model: nn.Module, data_loader: DataLoader) -> pd.DataFrame:
        """
        Return predict, radius
        """
        certify_model = Smooth(model, self.num_classes, self.sigma, self.N0, self.N, self.alpha, self.calculator.device)
        certify_results = []
        idx = 0
        certify_sample = 100

        for batch_id, batch_data in enumerate(data_loader):
            inputs, outputs = batch_data
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                if idx % certify_sample == 0:
                    input, output = inputs[i], outputs[i]
                    pred, radius = certify_model.certify(input)
                    correct = pred == output.data.max()
                    certify_result = {
                        "radius": radius,
                        "correct": correct
                    }
                    certify_results.append(certify_result)
                    idx += 1 
        return pd.DataFrame(certify_results)
    
    def accuracy_at_radii(self, model: nn.Module, data_loader: DataLoader, radii: np.ndarray) -> np.ndarray:
        certify_results = self.certify(model, data_loader)
        accuracy_calculator = ApproximateAccuracy(certify_results)
        return accuracy_calculator.at_radii(radii)

    def certify_train_radii(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        return self.accuracy_at_radii(model, data_loader, radii)

    def certify_test_radius(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(self.valid_data, batch_size=self.batch_size)
        return self.accuracy_at_radii(model, data_loader, radii)


class Server(BasicServer):
    def __init__(self, option, model: nn.Module, clients: list, test_data=None):
        super().__init__(option, model, clients, test_data)
        self.sigma = option['sigma_certify']
        self.N0 = option['n0']
        self.N = option['n']
        self.alpha = option['alpha_certify']

        self.num_classes = option['num_classes']
        self.radii = np.arange(0, 1.6, 0.1)
        self.batch_size = len(self.test_data) if option['batch_size']==-1 else option['batch_size']

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        
        logger.time_start('Total Time Cost')
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')

            # federated train
            self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)
            # self.certify()

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval):
                logger.log(self)
                if "certified_information" not in logger.output.keys():
                    logger.output['certified_information'] = {}
                output = self.log_certify()
                logger.output['certified_information'][f"round_{round}"] = output

            if round % self.option['log_interval'] == 0:
                if not os.path.exists(f"fedtask/{self.option['task']}/record/{self.option['session_name']}"):
                    os.makedirs(f"fedtask/{self.option['task']}/record/{self.option['session_name']}")
                logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
                torch.save(self.model.state_dict(), f"fedtask/{self.option['task']}/record/{self.option['session_name']}/model_{round}.pth")

        print("=================End==================")

        logger.time_end('Total Time Cost')
        # save results as .json file
        breakpoint()
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))

    def log_certify(self):
        server_certify_acc = self.certify().tolist()
        output = {}
        output["server_certify_acc"] = server_certify_acc
        output["client_certify_acc"] = {}

        for idx in range(self.num_clients):
            client_certify_acc = self.clients[idx].certify_test_radius(self.model, self.radii)
            output["client_certify_acc"][idx] = client_certify_acc.tolist()
        return output

    def certify(self):
        data_loader = self.calculator.get_data_loader(self.test_data, batch_size=self.batch_size)
        certify_model = Smooth(self.model, self.num_classes, self.sigma, self.N0, self.N, self.alpha, device=self.calculator.device)
        certify_results = []
        idx = 0
        certify_sample = 100

        for batch_id, batch_data in enumerate(data_loader):
            inputs, outputs = batch_data
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                if idx % certify_sample == 0:
                    input, output = inputs[i], outputs[i]
                    pred, radius = certify_model.certify(input)
                    correct = pred == output.data.max()
                    certify_result = {
                        "radius": radius,
                        "correct": correct
                    }
                    certify_results.append(certify_result)
                    idx += 1 
        df = pd.DataFrame(certify_results)
        
        # cal accuracy (certify accuracy)
        accuracy_calculator = ApproximateAccuracy(df)
        return accuracy_calculator.at_radii(self.radii)