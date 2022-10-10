#!/usr/bin/python
# !/usr/bin/env python

import collections
import math
import os
import time

import matplotlib.pyplot as plt
import xgboost as xgb

import tool
import xgboost_module as xgbm
from lru import run_lru
from other_algorithms import opt, deterministic_algorithm, random_pick
from xgboost_module import get_feature_vector

# This code implements the hybrid policy that keeps simulating deterministic only.
# At a check point, calculate the CR of the hybrid policy for the WHOLE HISTORY,
# If CR < log p, then run ML.
# Else, run deterministic.


def xgb_hybrid_picking2(model, window_size, sequence, k,
                       number_of_box_kinds, miss_cost,
                       check_length, adversarial=False):
    print('running xgb_hybrid_algorithm')

    _, _, det_req2box, det_req2counter, det_req2mi = deterministic_algorithm(sequence, k, number_of_box_kinds,
                                                                             miss_cost)

    deterministic_box_countings = [0 for _ in range(number_of_box_kinds)]
    deterministic_box_countings[0] = 1
    deterministic_current_box = 0
    pointer = 0
    total_impact = 0
    box_seq = []
    oracle_impact = 0
    run_oracle = True
    hybrid_global_lru = collections.OrderedDict()
    # deterministic_global_lru = collections.OrderedDict()
    seg = 0

    while pointer < len(sequence):
        start_pointer = pointer
        if run_oracle:
            if adversarial:
                box_seq.append(0)
                cache_size = k
                oracle_impact += 3 * cache_size * cache_size * miss_cost
            else:
                feature = xgb.DMatrix([get_feature_vector(sequence, pointer, window_size)])
                oracle = model.predict(feature)[0]
                box_seq.append(int(oracle))
                cache_size = k / (2 ** oracle)
                oracle_impact += 3 * cache_size * cache_size * miss_cost
        else:
            cache_size = k / (2 ** (number_of_box_kinds - deterministic_current_box - 1))
            box_seq.append(int(number_of_box_kinds - deterministic_current_box - 1))
            if deterministic_current_box == number_of_box_kinds - 1:
                deterministic_current_box = 0
            elif deterministic_box_countings[deterministic_current_box] % 4 == 0:
                deterministic_current_box = deterministic_current_box + 1
            else:
                deterministic_current_box = 0
            deterministic_box_countings[deterministic_current_box] = deterministic_box_countings[
                                                                         deterministic_current_box] + 1

        box_width = miss_cost * cache_size
        total_impact += 3 * cache_size * cache_size * miss_cost
        # Compartmentalization
        # Load top pages from LRU stack.
        mycache = collections.OrderedDict()
        for pid in hybrid_global_lru.keys():
            mycache[pid] = True
            mycache.move_to_end(pid, last=True)
            if len(mycache) == cache_size:
                break

        pointer = run_lru(mycache, cache_size, sequence, pointer, box_width, 1, miss_cost)
        # Update global stack
        for x in range(start_pointer, pointer):
            if sequence[x] in hybrid_global_lru.keys():
                hybrid_global_lru.move_to_end(sequence[x], last=False)
            else:
                hybrid_global_lru[sequence[x]] = True
                hybrid_global_lru.move_to_end(sequence[x], last=False)

        if int(pointer / check_length) > seg:
            seg = int(pointer / check_length)
            opt_temp, _, _ = opt(sequence[0:pointer], k, number_of_box_kinds, miss_cost)
            if run_oracle and (total_impact / opt_temp) > number_of_box_kinds:
                deterministic_current_box = det_req2box[pointer]
                deterministic_box_countings = det_req2counter[pointer]
            run_oracle = (total_impact / opt_temp) <= number_of_box_kinds

    return total_impact, oracle_impact / total_impact, box_seq


k = 64
number_of_box_kinds = 7
window_size = 256
miss_cost = 100
check_length = 10000

# model parameters
max_depth = 5
params = {
    'objective': 'multi:softmax',
    'eta': 0.1,
    'max_depth': max_depth,
    'num_class': number_of_box_kinds
}
num_round = 10  # training rounds
model_filename = 'xgb_model_k{0}_b{1}_w{2}_s{3}_d{4}_r{5}'.format(k, number_of_box_kinds, window_size,
                                                                  miss_cost, max_depth, num_round)
time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
os.makedirs(time_str + '-result-plots-hybrid2')

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
det_array = []
rand_array = []
oracle_array = []
xgb_hybrid_array = []
sharing_ratio = []
tick = 1
for trace_name in names:
    # trace_name = 'bzip'
    print('++++++++++ running', tick, '/', len(names), '++++++++++')
    tick += 1
    # train_seq = tool.read_crc_seq('datasets/' + trace_name + '_train.csv')
    test_seq = tool.read_crc_seq('datasets/' + trace_name + '_test.csv')

    ##########################
    trace_name = 'adversary'
    test_seq = [i for i in range(60000)]
    ##########################
    '''

    print('########### testing xgb ###########')
    xgb_impact, xgb_boxes, _ = xgbm.test_xgboost(model, test_seq, k, miss_cost, window_size)
    print('########### testing hybrid alg ###########')
    xgb_hybrid_impact, sr, xgb_m_boxes = xgb_hybrid_picking2(model, window_size, test_seq, k, number_of_box_kinds,
                                                             miss_cost, check_length)

    '''
    xgb_impact = 3*k*k*miss_cost*math.ceil(len(test_seq)/k)
    xgb_hybrid_impact, sr, xgb_m_boxes = xgb_hybrid_picking2(model, window_size, test_seq, k, number_of_box_kinds,
                                                             miss_cost, check_length, True)
    #'''
    sharing_ratio.append(sr)

    print('######### testing other methods ############')
    opt_impact, opt_boxes, _ = opt(test_seq, k, number_of_box_kinds, miss_cost)
    det_impact, det_boxes, _, _, _ = deterministic_algorithm(test_seq, k, number_of_box_kinds, miss_cost)
    random_impact = random_pick(test_seq, k, number_of_box_kinds, miss_cost)

    print('oracle:       ', int(xgb_impact))
    oracle_array.append(int(xgb_impact))
    print('opt:          ', int(opt_impact))
    opt_array.append(int(opt_impact))
    print('random:       ', int(random_impact))
    rand_array.append(int(random_impact))
    print('deterministic:', int(det_impact))
    det_array.append(int(det_impact))
    print('xgb-hybrid:   ', int(xgb_hybrid_impact))
    xgb_hybrid_array.append(int(xgb_hybrid_impact))

    # draw plot
    h = ['Random', 'deterministic', 'Oracle', 'hybrid']
    v = [random_impact / opt_impact, det_impact / opt_impact, xgb_impact / opt_impact,
         xgb_hybrid_impact / opt_impact]
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
    plt.savefig(r'' + time_str + '-result-plots-hybrid2/1-v-all-s' + str(miss_cost) + 'd' + str(params['max_depth']) + 'r' +
                str(num_round) + '-' + trace_name + '.jpg')
    plt.close()
    break

print("...........................................................")
print('1-v-all-s' + str(miss_cost) + 'd' + str(params['max_depth']) + 'r' + str(num_round))
print("opt")
print(opt_array)
print('random')
print(rand_array)
print('deterministic')
print(det_array)
print('oracle')
print(oracle_array)
print('hybrid')
print(xgb_hybrid_array)
print('sharing ratio')
print(sharing_ratio)
