import os, sys, time, json
import numpy as np
import time
import utils
from utils import recorder

from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer, BaseTrainer 
import evaluation
from utils import check_convention, config_path_prefix
import argparse
import plot

DEFAULT_RES_SAVE_PATH_PREFIX = "./res_pred"

def train_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader,unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader,test_loader)
    eval_res = trainer.final_eval(test_loader)
    
    start_eval_time = time.time()
    pred_all, y_all = trainer.test(all_loader)
    end_eval_time = time.time()
    eval_time = end_eval_time - start_eval_time
    print("eval time is %s" % eval_time) 
    recorder.record_time(eval_time)
    pred_matrix = dataloader.reconstruct_pred(pred_all)


    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    recorder.record_pred(pred_matrix)
    recorder.to_file(param['path_res'])

    #4. plot
    rawdata, TR, TE = dataloader.get_data()
    plot.plot_all(pred_matrix, TR, TE, param['path_pic'])

    return recorder

def train_convention_by_param(param):
    #0. recorder reset防止污染数据
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    trainX, trainY, testX, testY, allX = dataloader.generate_torch_dataset() 

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(trainX, trainY)
    eval_res = trainer.final_eval(testX, testY)
    pred_all = trainer.test(allX)
    pred_matrix = dataloader.reconstruct_pred(pred_all)

    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    recorder.record_pred(pred_matrix)

    return recorder 




include_path = [
    'indian_transformer.json',
    'pavia_transformer.json',
    'WH_transformer.json',
]

def run_all():
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        convention = check_convention(name)
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        uniq_name = param.get('uniq_name', name)
        print('start to train %s...' % uniq_name)
        time_stamp = str(int(time.time()))
        path = '%s/%s_%s' % (save_path_prefix, uniq_name, time_stamp) 
        path_pic = '%s/%s_%s.png' % (save_path_prefix, uniq_name, time_stamp) 
        param['path_res'] = path
        param['path_pic'] = path_pic
        if convention:
            train_convention_by_param(param)
        else:
            train_by_param(param)
        print('model eval done of %s...' % uniq_name)



def result_file_exists(prefix, file_name_part):
    ll = os.listdir(prefix)
    for l in ll:
        if file_name_part in l:
            return True
    return False

def run_one_multi_times(json_str, ori_uniq_name):
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    times = 5
    for i in range(times): 
        uniq_name = '%s_%s' % (ori_uniq_name, i)
        if result_file_exists(DEFAULT_RES_SAVE_PATH_PREFIX, uniq_name):
            print('%s has been run. skip...' % uniq_name)
            continue
        path = '%s/%s' % (save_path_prefix, uniq_name) 
        json_str['path_res'] = path
        print('start to train %s...' % uniq_name)
        train_by_param(json_str)
        print(json_str)
        print('model eval done of %s...' % uniq_name)

if __name__ == "__main__":
    run_all()
    

