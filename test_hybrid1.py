#!/usr/bin/python
# !/usr/bin/env python

from xgb_hybrid_1 import xgb_hybrid_picking1, xgb_hybrid_picking1_version_2, xgbm
import tool


def main():
    k = 64
    number_of_box_kinds = 7
    window_size = 256
    miss_cost = 100
    check_length = 10000

    # model parameters
    max_depth = 5
    num_round = 10  # training rounds
    model_filename = 'xgb_model_k{0}_b{1}_w{2}_s{3}_d{4}_r{5}'.format(k, number_of_box_kinds,
                                                                      window_size,
                                                                      miss_cost, max_depth, num_round)

    names = ['astar', 'bwaves', 'bzip', 'cactusadm',
             'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
             'milc', 'omnetpp', 'sphinx3', 'xalanc']

    print('\n\n')
    found, model = xgbm.check_and_load_model(model_filename)
    if not found:
        return

    array112 = []
    array121 = []
    array122 = []
    tick = 1
    for trace_name in names:
        print('++++++++++ running', tick, '/', len(names), '++++++++++')
        tick += 1
        test_seq = tool.read_crc_seq('datasets/' + trace_name + '_test.csv')
        print('########### 112 ###########')
        mi, _, _ = xgb_hybrid_picking1(2, model, window_size, test_seq,
                                       k, number_of_box_kinds,
                                       miss_cost, check_length)
        array112.append(mi)
        print('########### 121 ###########')
        mi, _, _ = xgb_hybrid_picking1_version_2(1, model, window_size, test_seq,
                                                 k, number_of_box_kinds,
                                                 miss_cost, check_length)
        array121.append(mi)
        print('########### 122 ###########')
        mi, _, _ = xgb_hybrid_picking1_version_2(2, model, window_size, test_seq,
                                                 k, number_of_box_kinds,
                                                 miss_cost, check_length)
        array122.append(mi)

    # check adversary
    print('++++++++++ running adversary ++++++++++')
    test_seq = [i for i in range(60000)]
    print('########### 112 ###########')
    mi, _, _ = xgb_hybrid_picking1(2, model, window_size, test_seq,
                                   k, number_of_box_kinds,
                                   miss_cost, check_length,
                                   adversarial=True)
    array112.append(mi)
    print('########### 121 ###########')
    mi, _, _ = xgb_hybrid_picking1_version_2(1, model, window_size, test_seq,
                                             k, number_of_box_kinds,
                                             miss_cost, check_length,
                                             adversarial=True)
    array121.append(mi)
    print('########### 122 ###########')
    mi, _, _ = xgb_hybrid_picking1_version_2(2, model, window_size, test_seq,
                                             k, number_of_box_kinds,
                                             miss_cost, check_length,
                                             adversarial=True)
    array122.append(mi)

    print("...........................................................")
    print(array112)
    print(array121)
    print(array122)


main()
