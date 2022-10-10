#!/usr/bin/python
# !/usr/bin/env python


import random

# 1-v-all-s1000d5r5
opt = [4434000, 3858000, 13005000, 49212000, 32211000, 12324000,
       80913000, 225000, 12075000, 3588000, 198897000, 38232000, 20571000]
rand = [58827744.0, 36795954.0, 59681712.0, 80103282.0, 100006560.0, 50938941.0, 180774915.0,
        32198913.0, 122272770.0, 38622486.0, 1084806786.0, 90914769.0, 469657860.0]
deterministic = [9024000.0, 18432000.0, 26904000.0, 194304000.0, 172032000.0, 55296000.0,
                 258048000.0, 363000.0, 27366000.0, 15360000.0, 608550000.0, 99888000.0, 55350000.0]
oracle = [17487000.0, 4173000.0, 19323000.0, 81399000.0, 156444000.0, 15867000.0, 148611000.0,
          225000.0, 83520000.0, 5358000.0, 199434000.0, 88653000.0, 22386000.0]

idx = [i for i in range(len(opt))]

mla = []
for tick in range(10 ** 6):
    print(tick)
    random.shuffle(idx)

    deterministic_mi = 0
    oracle_mi = 0
    opt_mi = 0
    mla_mi = 0

    for i in range(len(idx)):
        if i == 0:
            mla_mi += oracle[idx[i]]
        else:
            if mla_mi > deterministic_mi:
                # if mla_mi / opt_mi > 8:
                mla_mi += deterministic[idx[i]]
            else:
                mla_mi += oracle[idx[i]]

        deterministic_mi += deterministic[idx[i]]
        oracle_mi += oracle[idx[i]]
        opt_mi += opt[idx[i]]

    mla.append(mla_mi)

print("deterministic's competitive ratio: " + str(sum(deterministic) / sum(opt)))
print("Pure oracle's competitive ratio: " + str(sum(oracle) / sum(opt)))
print("MLA competitive ratio in average: " + str(sum(mla) / len(mla) / sum(opt)))
print("MLA max competitive ratio: " + str(max(mla) / sum(opt)))
print("MLA<deterministic: " + str(sum([v < sum(deterministic) for v in mla]) / len(mla)))
print("MLA<deterministic/1.5: " + str(sum([v < (sum(deterministic) / 1.5) for v in mla]) / len(mla)))
