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
        self.images_all_vlds = [torch.unsqueeze(self.valid_data[i][0], dim=0) for i in range(len(self.valid_data))]
        self.labels_all_vlds = [self.valid_data[i][1].item() for i in range(len(self.valid_data))]
        self.set_label = list(set(self.labels_all_vlds))
        self.num_classes = len(self.set_label)
        self.channel, self.im_size_1, self.im_size_2 = self.valid_data[0][0].shape
        self.indices_class_vlds = [[] for c in range(10)]
        for i, lab in enumerate(self.labels_all_vlds):
            self.indices_class_vlds[lab].append(i)
        self.image_syn = torch.randn(size=(self.num_classes*option['ipc'], self.channel, self.im_size_1, self.im_size_2), dtype=torch.float, requires_grad=True, device=option['device'])
        self.label_syn = torch.tensor([np.ones(option['ipc'])*i for i in self.set_label], dtype=torch.long, requires_grad=False).view(-1).to(option['device']) # [0,0,0, 1,1,1, ..., 9,9,9]
        self.lr_img = option['lr_img']
        self.images_all_vlds = torch.cat(self.images_all_vlds, dim=0).to(option['device'])
        self.labels_all_vlds = torch.tensor(self.labels_all_vlds, dtype=torch.long, device=option['device'])

    def get_images(self,c, n): # get random n images from class c
            try:
                idx_shuffle = np.random.permutation(self.indices_class_vlds[c])[:n]
                return self.images_all_vlds[idx_shuffle]
            except:
                breakpoint()
    
    def distance_wb(self,gwr, gws):
        shape = gwr.shape
        if len(shape) == 4: # conv, out*in*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:  # layernorm, C*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2])
            gws = gws.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2: # linear, out*in
            tmp = 'do nothing'
        elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            return torch.tensor(0, dtype=torch.float, device=gwr.device)

        dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
        dis = dis_weight
        return dis

    def match_loss(self,gw_syn, gw_real, option):
        dis = torch.tensor(0.0).to(option['device'])

        if option['dis_metric'] == 'ours':
            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                dis += self.distance_wb(gwr, gws)

        elif option['dis_metric'] == 'mse':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

        elif option['dis_metric'] == 'cos':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

        else:
            exit('unknown distance function: %s'%option['dis_metric'])

        return dis
    def get_loops(self,ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
        if ipc == 1:
            outer_loop, inner_loop = 1, 1
        elif ipc == 10:
            outer_loop, inner_loop = 10, 50
        elif ipc == 20:
            outer_loop, inner_loop = 20, 25
        elif ipc == 30:
            outer_loop, inner_loop = 30, 20
        elif ipc == 40:
            outer_loop, inner_loop = 40, 15
        elif ipc == 50:
            outer_loop, inner_loop = 50, 10
        else:
            outer_loop, inner_loop = 0, 0
            exit('loop hyper-parameters are not defined for %d ipc'%ipc)
        return outer_loop, inner_loop
    
    def update_syn_imgs(self, model):
        import copy
        outer_loop, inner_loop = self.get_loops(self.option['ipc'])
        optimizer_img = torch.optim.SGD([self.image_syn], lr=self.option['lr_img'], momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = criterion = nn.CrossEntropyLoss().to(self.option['device'])
        syn_model = copy.deepcopy(model)
        loss_avg = 0
        for ol in range(outer_loop):
            loss  =0
            for i, c in enumerate(self.set_label):
                # if len(self.indices_class_vlds[c]) != 0:
                try:
                    batch_size = self.option['batch_size'] if self.option['batch_size'] <= len(self.indices_class_vlds[c]) else len(self.indices_class_vlds[c])
                    img_real = self.get_images(c, batch_size)
                    lab_real = torch.ones((img_real.shape[0],), device=self.option['device'], dtype=torch.long) * c
                    img_syn = self.image_syn[i*self.option['ipc']:(i+1)*self.option['ipc']].reshape((self.option['ipc'], self.channel, self.im_size_1, self.im_size_2))
                    lab_syn = torch.ones((self.option['ipc'],), device=self.option['device'], dtype=torch.long) * c
                    net_parameters = list(syn_model.parameters())
                    output_real = syn_model(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = syn_model(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    loss += self.match_loss(gw_syn, gw_real, self.option)
                except:
                    breakpoint()
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
        loss_avg /= (self.num_classes*outer_loop)
        return loss_avg
    
    def train(self, model: nn.Module):
        """
        Training process for smoothed classifier
        Client training with noisy data
        """
        syn_loss = self.update_syn_imgs(model)
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        syn_optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
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
        certify_sample = 1

        for batch_id, batch_data in enumerate(data_loader):
            inputs, outputs = batch_data
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                if idx % certify_sample == 0:
                    input, output = inputs[i], outputs[i]
                    pred, radius = certify_model.certify(input)
                    correct = (pred == output.data.max()).item()
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
        return accuracy_calculator.at_radii(radii), certify_results

    def certify_train_radii(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        return self.accuracy_at_radii(model, data_loader, radii)

    def certify_test_radius(self, model: nn.Module, radii: np.ndarray):
        data_loader = self.calculator.get_data_loader(self.valid_data, batch_size=self.batch_size,shuffle=False)
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
                
            if round % self.option['log_interval'] == 0 and round != 0:
                if "certified_information" not in logger.output.keys():
                    logger.output['certified_information'] = {}
                output = self.log_certify()
                logger.output['certified_information'][f"round_{round}"] = output

                if not os.path.exists(f"fedtask/{self.option['task']}/record/{self.option['session_name']}"):
                    os.makedirs(f"fedtask/{self.option['task']}/record/{self.option['session_name']}")
                logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
                torch.save(self.model.state_dict(), f"fedtask/{self.option['task']}/record/{self.option['session_name']}/model_{round}.pth")

        print("=================End==================")

        logger.time_end('Total Time Cost')
        # save results as .json file
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))

    def log_certify(self):
        list_syn_imgs = []
        list_syn_label = []
        output = {}
        for idx in range(self.num_clients):
            list_syn_imgs.append(self.clients[idx].image_syn)
            list_syn_label.append(self.clients[idx].label_syn)
        all_syn_imgs = torch.cat(list_syn_imgs,dim=0)
        all_syn_label = torch.cat(list_syn_label,dim=0)
        output["client_lable"] =  all_syn_label.tolist()
        syn_client_certify_acc, syn_df_acc = self.certify_(all_syn_imgs,all_syn_label)
        output["client_certify_acc"] = syn_client_certify_acc.tolist()
        output["client_certify_acc_samples"] = syn_df_acc.values.tolist()
        # breakpoint()
        server_certify_acc, df_acc = self.certify()
        output["server_certify_acc"] = server_certify_acc.tolist()
        output["server_certify_acc_samples"] = df_acc.values.tolist()

        return output
    
    def certify_(self,images, labels):
        certify_model = Smooth(self.model, self.num_classes, self.sigma, self.N0, self.N, self.alpha, device=self.calculator.device)
        certify_results = []

        for i in range(images.shape[0]):
            input = images[i]
            output = labels[i]
            pred, radius = certify_model.certify(input)
            correct = (pred == output.data.max()).item()
            certify_result = {
                        "radius": radius,
                        "correct": correct
                    }
            certify_results.append(certify_result)
        df = pd.DataFrame(certify_results)
        # cal accuracy (certify accuracy)
        accuracy_calculator = ApproximateAccuracy(df)
        return accuracy_calculator.at_radii(self.radii), df


    def certify(self):
        data_loader = self.calculator.get_data_loader(self.test_data, batch_size=self.batch_size,shuffle=False)
        certify_model = Smooth(self.model, self.num_classes, self.sigma, self.N0, self.N, self.alpha, device=self.calculator.device)
        certify_results = []
        idx = 0
        certify_sample = 1

        for batch_id, batch_data in enumerate(data_loader):
            inputs, outputs = batch_data
            batch_size = inputs.shape[0]
            
            for i in range(batch_size):
                if idx % certify_sample == 0:
                    input, output = inputs[i], outputs[i]
                    pred, radius = certify_model.certify(input)
                    correct = (pred == output.data.max()).item()
                    certify_result = {
                        "radius": radius,
                        "correct": correct
                    }
                    certify_results.append(certify_result)
                idx += 1 
        df = pd.DataFrame(certify_results)
        # cal accuracy (certify accuracy)
        accuracy_calculator = ApproximateAccuracy(df)
        return accuracy_calculator.at_radii(self.radii), df
