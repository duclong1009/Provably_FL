import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from numpy.linalg import norm
import argparse
def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', help='name of nound;', type=str, default='mnist')
    parser.add_argument("--task_name",type=str)
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option


def plot_line(x,y1,y2, append=""):
    plt.plot(x,y1,label="Server",)
    plt.plot(x,y2, label="Aggregated Client")
    plt.xlabel("Radius")
    plt.ylabel("Certified Accuracy")
    plt.legend()
    plt.savefig(f"{saved_path}/aggregated_{append}_{round}.jpg")
    plt.clf()

def cal_cosine_sim(x,y):
    if norm(x) == 0 and norm(y) == 0:
        return 1
    elif norm(y) == 0:
        return 0 
    return np.dot(x,y)/(norm(x) * norm(y))

def at_radii(certify_results, radii: np.ndarray) -> np.ndarray:
        return np.array([at_radius(certify_results, radius) for radius in radii])

def at_radius( df: pd.DataFrame, radius: float):
    # True: predict == gt & cert_radis >= radius
    return (df["correct"] & (df["radius"] >= radius)).mean()

def box_plot_score_on_each_class(fi):
    plt.boxplot(fi)
    plt.xlabel("Class")
    plt.ylabel("Radius")
    plt.savefig(f"{saved_path}/box_plot_scored_on_each_class_{round}.jpg")
    plt.clf()

def hist_for_each_class(fi,bins=30):
    fig, axs = plt.subplots(2,5, sharex=True, sharey=True)

    for i in range(10):
        axs[i//5,i%5].hist(fi[i],bins=bins)
        axs[i//5,i%5].set_title(f"Class {i}")
    plt.savefig(f"{saved_path}/distribution_scored_on_each_class_{round}.jpg")
    plt.clf()

def find_index(list_to_check, value_to_check):
    list_idx = []
    for idx, value in enumerate(list_to_check):
        if value == value_to_check:
            list_idx.append(idx)
    return list_idx

def plot_cosine_similarity(list_cosine):
    x = range(0,len(list_cosine))
    y = list_cosine
    plt.figure(figsize=(20,5))
    plt.plot(x,y)
    plt.grid(which='both')
    plt.xticks(x)
    plt.savefig(f"{saved_path}/cosine_similarity_{round}.jpg")
    plt.clf()
    # fig = plt.figure()

option = read_option()
# name_session = "cifar10_cnum100_pareto_fedprob_sigma0.5"
name_session = option['task_name']
path = f"../fedtask/cifar10_cnum100_pareto/record/{name_session}.json"
with open(path, "r") as f:
    data = json.load(f)

with open("../fedtask/cifar10_cnum100_pareto/data.json","r") as f:
    data_information = json.load(f)

#config
radii = np.arange(0, 1.6, 0.1)

for round in range(40,520,40):
    certified_acc_dict = {}
    certified_acc_list = []
    score_dict = {}
    list_n_samples = []
    for label in range(10):
        score_dict[label] = []

    #cal certified server 
    sever_infor = data['certified_information'][f'round_{round}']["server_certify_acc_samples"]
    sever_certied_acc = at_radii(pd.DataFrame(sever_infor,columns=['radius','correct']), radii)

    for i,client_name in enumerate(data_information['client_names']):
        client_information = data_information[client_name]
        list_label = client_information['dvalid']['y']
        certi_information = data['certified_information'][f'round_{round}']['client_certify_acc_samples'][str(i)]
        rs = at_radii(pd.DataFrame(certi_information,columns=['radius','correct']), radii)
        certified_acc_dict[i] = rs
        certified_acc_list.append(rs)
        list_n_samples.append(len(list_label))
        for ii, lb in enumerate(list_label):
            score_dict[lb].append(certi_information[ii])
    normed_rate = np.array(list_n_samples)/sum(list_n_samples)
    certied_acc_arr = np.array(certified_acc_list)
    aggregated_client_acc = np.matmul(normed_rate.T, certied_acc_arr)
    list_cosine_sim_score = []
    for i in range(100):
        list_cosine_sim_score.append(cal_cosine_sim(sever_certied_acc, certified_acc_dict[i]))
            


    # v2_certified_acc_dict = {}
    # v2_certified_acc_list = []
    # for radius in radii:
    #     list_c = []
    #     for i,client_name in enumerate(data_information['client_names']):
    #         list_acc = []
    #         client_information = data_information[client_name]
    #         list_label = client_information['dvalid']['y']
    #         certi_information = data['certified_information'][f'round_{round}']['client_certify_acc_samples'][str(i)]
    #         df_infor = pd.DataFrame(certi_information,columns=['radius','correct'])
        
    #         list_score = []
    #         for lb in range(10):
    #             if lb in list_label:
    #                 idx = find_index(list_label,lb)
    #                 class_df = df_infor.iloc[idx]
    #                 list_score.append((class_df["correct"] & (class_df["radius"] >= radius)).mean())
    #             else:
    #                 list_score.append(0)
        
    #         list_c.append(list_score)
    #     breakpoint()
            # v2_certified_acc_dict[i] = list_acc
            # v2_certified_acc_list.append(list_acc)
            # print(list_acc)
    # v2_aggregated_client_acc = np.array(v2_certified_acc_list).mean(0)
    
    fi = []
    for key in score_dict.keys():
        fi.append(list(np.array(score_dict[key])[:,0]))

    saved_path = f"plot/image/{name_session}/"

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # plot_line(radii,sever_certied_acc,v2_aggregated_client_acc, "v2")
    plot_line(radii,sever_certied_acc,aggregated_client_acc)
    plot_cosine_similarity(list_cosine=list_cosine_sim_score)
    box_plot_score_on_each_class(fi)
    hist_for_each_class(fi)
