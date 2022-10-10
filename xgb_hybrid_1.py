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


# This code implements the hybrid policy that simulates both ML & deterministic.
# At a check point, if ML is better in the WHOLE HISTORY, then run ML.
# Else, run deterministic.


def xgb_hybrid_picking1(alpha, model, window_size, sequence, k,
                        number_of_box_kinds, miss_cost,
                        check_length, adversarial=False):
    print('running xgb_hybrid_algorithm 1.1')

    _, _, det_req2box, det_req2counter, det_req2mi = deterministic_algorithm(sequence, k,
                                                                             number_of_box_kinds,
                                                                             miss_cost)
    xgb_req2mi = []
    if not adversarial:
        _, _, ttt = xgbm.test_xgboost(model, sequence, k, miss_cost, window_size)
        for tt in ttt:
            xgb_req2mi.append(tt)

    det_box_counting = [0 for _ in range(number_of_box_kinds)]
    det_box_counting[0] = 1
    det_current_box = 0
    pointer = 0
    total_impact = 0
    box_seq = []
    oracle_impact = 0
    run_oracle = True
    hybrid_global_lru = collections.OrderedDict()
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
            cache_size = k / (2 ** (number_of_box_kinds - det_current_box - 1))
            box_seq.append(int(number_of_box_kinds - det_current_box - 1))
            if det_current_box == number_of_box_kinds - 1:
                det_current_box = 0
            elif det_box_counting[det_current_box] % 4 == 0:
                det_current_box = det_current_box + 1
            else:
                det_current_box = 0
            det_box_counting[det_current_box] = det_box_counting[
                                                    det_current_box] + 1

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
            if not adversarial:
                if run_oracle and xgb_req2mi[pointer - 1] > alpha * det_req2mi[pointer - 1]:
                    det_current_box = det_req2box[pointer]
                    det_box_counting = det_req2counter[pointer]
                run_oracle = xgb_req2mi[pointer - 1] <= alpha * det_req2mi[pointer - 1]
            else:
                mlmi = 3 * k * k * miss_cost * math.ceil(pointer / k)
                if run_oracle and mlmi > alpha * det_req2mi[pointer - 1]:
                    det_current_box = det_req2box[pointer]
                    det_box_counting = det_req2counter[pointer]
                run_oracle = mlmi <= alpha * det_req2mi[pointer - 1]

    return total_impact, oracle_impact / total_impact, box_seq


def xgb_hybrid_picking1_version_2(alpha, model, window_size, sequence, k,
                                  number_of_box_kinds, miss_cost,
                                  check_length, adversarial=False):
    print('running xgb_hybrid_algorithm 1.2')

    _, _, det_req2box, det_req2counter, det_req2mi = deterministic_algorithm(sequence, k,
                                                                             number_of_box_kinds,
                                                                             miss_cost)
    xgb_req2mi = []
    if not adversarial:
        _, _, ttt = xgbm.test_xgboost(model, sequence, k, miss_cost, window_size)
        for tt in ttt:
            xgb_req2mi.append(tt)

    det_box_counting = [0 for _ in range(number_of_box_kinds)]
    det_box_counting[0] = 1
    det_current_box = 0
    pointer = 0
    total_impact = 0
    box_seq = []
    oracle_impact = 0
    run_oracle = True
    hybrid_global_lru = collections.OrderedDict()
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
            cache_size = k / (2 ** (number_of_box_kinds - det_current_box - 1))
            box_seq.append(int(number_of_box_kinds - det_current_box - 1))
            if det_current_box == number_of_box_kinds - 1:
                det_current_box = 0
            elif det_box_counting[det_current_box] % 4 == 0:
                det_current_box = det_current_box + 1
            else:
                det_current_box = 0
            det_box_counting[det_current_box] = det_box_counting[
                                                    det_current_box] + 1

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
            if not adversarial:
                if run_oracle and xgb_req2mi[pointer - 1] > alpha * det_req2mi[pointer - 1]:
                    det_current_box = det_req2box[pointer]
                    det_box_counting = det_req2counter[pointer]

                if run_oracle:
                    run_oracle = xgb_req2mi[pointer - 1] <= alpha * det_req2mi[pointer - 1]
                else:
                    run_oracle = alpha * xgb_req2mi[pointer - 1] < det_req2mi[pointer - 1]
            else:
                mlmi = 3 * k * k * miss_cost * math.ceil(pointer / k)
                if run_oracle and mlmi > alpha * det_req2mi[pointer - 1]:
                    det_current_box = det_req2box[pointer]
                    det_box_counting = det_req2counter[pointer]

                if run_oracle:
                    run_oracle = mlmi <= alpha * det_req2mi[pointer - 1]
                else:
                    run_oracle = alpha * mlmi < det_req2mi[pointer - 1]

    return total_impact, oracle_impact / total_impact, box_seq


def main():
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
    model_filename = 'xgb_model_k{0}_b{1}_w{2}_s{3}_d{4}_r{5}'.format(k, number_of_box_kinds,
                                                                      window_size,
                                                                      miss_cost, max_depth, num_round)
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = time_str + '-result-plots-hybrid1'
    os.makedirs(result_dir)

    names = ['astar', 'bwaves', 'bzip', 'cactusadm',
             'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
             'milc', 'omnetpp', 'sphinx3', 'xalanc']

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
        print('++++++++++ running', tick, '/', len(names), '++++++++++')
        tick += 1
        test_seq = tool.read_crc_seq('datasets/' + trace_name + '_test.csv')

        print('########### testing xgb ###########')
        xgb_impact, xgb_boxes, _ = xgbm.test_xgboost(model, test_seq, k, miss_cost, window_size)

        print('########### testing hybrid alg ###########')

        xgb_hybrid_impact, sr, xgb_m_boxes = xgb_hybrid_picking1(1, model, window_size, test_seq,
                                                                 k, number_of_box_kinds,
                                                                 miss_cost, check_length)
        sharing_ratio.append(sr)

        print('######### testing other methods ############')
        opt_impact, opt_boxes, _ = opt(test_seq, k, number_of_box_kinds, miss_cost)
        det_impact, det_boxes, _, _, _ = deterministic_algorithm(test_seq, k, number_of_box_kinds,
                                                                 miss_cost)
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
        plt.title('Memory impact on the test trace of ' + trace_name,
                  fontdict={'weight': 'black'})
        for index, value in enumerate(v):
            plt.text(index - 0.07, value + 0.1, str(round(value, 2)),
                     fontdict={'weight': 'black'})
        # plt.show()
        plt.savefig(r'' + result_dir + '/' + trace_name + '.jpg')
        plt.close()

    # check adversary
    print('++++++++++ running adversary ++++++++++')
    trace_name = 'adversary'
    test_seq = [i for i in range(60000)]
    xgb_impact = 3 * k * k * miss_cost * math.ceil(len(test_seq) / k)
    xgb_hybrid_impact, sr, xgb_m_boxes = xgb_hybrid_picking1(1, model, window_size, test_seq,
                                                             k, number_of_box_kinds,
                                                             miss_cost, check_length,
                                                             adversarial=True)
    sharing_ratio.append(sr)

    print('######### testing other methods ############')
    opt_impact, opt_boxes, _ = opt(test_seq, k, number_of_box_kinds, miss_cost)
    det_impact, det_boxes, _, _, _ = deterministic_algorithm(test_seq, k, number_of_box_kinds,
                                                             miss_cost)
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
    plt.title('Memory impact on the test trace of ' + trace_name,
              fontdict={'weight': 'black'})
    for index, value in enumerate(v):
        plt.text(index - 0.07, value + 0.1, str(round(value, 2)),
                 fontdict={'weight': 'black'})
    # plt.show()
    plt.savefig(r'' + result_dir + '/' + trace_name + '.jpg')
    plt.close()

    print("...........................................................")
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


# main()
