#!/usr/bin/python
# !/usr/bin/env python


import os
import time

import matplotlib.pyplot as plt

import tool
import xgboost_module as xgbm
from other_algorithms import opt, deterministic_algorithm, random_pick

k = 2 ** 7
number_of_box_kinds = 2
window_size = 256
miss_cost = 100

# model parameters
max_depth = 90
params = {
    'objective': 'multi:softmax',
    'eta': 0.1,
    'max_depth': max_depth,
    'num_class': number_of_box_kinds
}
num_round = 99  # training rounds
model_filename = 'xgb_model_k{0}_b{1}_w{2}_s{3}_d{4}_r{5}'.format(k, number_of_box_kinds, window_size,
                                                                  miss_cost, max_depth, num_round)

time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
os.makedirs(time_str + '-result-plots')

files = os.listdir('datasets')
# print(files)
names = {}
for f in files:
    # assert that the file is a csv
    if f[-4:] == ".csv":
        temp = f.split('_')[0]
        names[temp] = 0

print('\n\n')
found, model = xgbm.check_and_load_model(model_filename)
if not found:
    print('########### building train set ###########')
    train_x = []
    train_y = []
    for trace_name in names:
        begin = time.time()
        print('building features for ' + trace_name)
        train_seq = tool.read_crc_seq('datasets/' + trace_name + '_train.csv')
        x, y = xgbm.features_and_labels(train_seq, k, number_of_box_kinds, miss_cost, window_size)
        for i in range(len(x)):
            train_x.append(x[i])
            train_y.append(y[i])
        opt_latency = time.time() - begin
        print("Latency of OPT: ", opt_latency)

    print('########### training xgb ###########')
    begin = time.time()
    model = xgbm.train_xgboost(train_x, train_y, params, num_round)
    print("train latency = ", time.time() - begin)
    xgbm.save_model(model, model_filename)  # Save this model for later
else:
    # We have a model to use, so no need to train
    print('Found a previous model to use!', 'saved_models/', model_filename)
    print('No training necessary :D')

opt_array = []
deterministic_alg_array = []
rand_array = []
oracle_array = []
tick = 1
for trace_name in names:
    # trace_name = 'bzip'
    print('++++++++++ running', tick, '/', len(names), '++++++++++')
    tick += 1
    # train_seq = tool.read_crc_seq('datasets/' + trace_name + '_train.csv')
    test_seq = tool.read_crc_seq('datasets/' + trace_name + '_test.csv')

    print('########### testing xgb ###########')
    xgb_impact, _, _ = xgbm.test_xgboost(model, test_seq, k, miss_cost, window_size)
    print('######### testing other methods ############')
    opt_impact, _, _ = opt(test_seq, k, number_of_box_kinds, miss_cost)
    deterministic_alg_impact, _, _, _, _ = deterministic_algorithm(test_seq, k, number_of_box_kinds, miss_cost)
    random_impact = random_pick(test_seq, k, number_of_box_kinds, miss_cost)

    print('oracle:           ', int(xgb_impact))
    print('opt:              ', int(opt_impact))
    print('random:           ', int(random_impact))
    print('deterministic_alg:', int(deterministic_alg_impact))
    oracle_array.append(int(xgb_impact))
    opt_array.append(int(opt_impact))
    rand_array.append(int(random_impact))
    deterministic_alg_array.append(int(deterministic_alg_impact))

    # draw plot
    h = ['Random', 'deterministic_alg', 'Oracle']
    v = [random_impact / opt_impact, deterministic_alg_impact / opt_impact, xgb_impact / opt_impact]
    plt.bar(h, v, width=0.3, color='green')
    plt.xlabel('methods', fontdict={'weight': 'black'})
    plt.grid(axis='y', linestyle='-.', linewidth=1, color='black', alpha=0.5)
    plt.ylabel('competitive ratio on memory impact', fontdict={'weight': 'black'})
    plt.ylim(0, max(v) * 1.1)
    plt.title('Memory impact on the test trace of ' + trace_name + ', s=' + str(miss_cost),
              fontdict={'weight': 'black'})
    for index, value in enumerate(v):
        plt.text(index - 0.07, value + 0.1, str(round(value, 2)),
                 fontdict={'weight': 'black'})
    # plt.show()
    plt.savefig(r'' + time_str + '-result-plots/1-v-all-s' + str(miss_cost) + 'd' + str(params['max_depth']) + 'r' +
                str(num_round) + '-' + trace_name + '.jpg')
    plt.close()

print("...........................................................")
print('1-v-all-s' + str(miss_cost) + 'd' + str(params['max_depth']) + 'r' + str(num_round))
print("opt")
print(opt_array)
print('random')
print(rand_array)
print('deterministic')
print(deterministic_alg_array)
print('oracle')
print(oracle_array)
