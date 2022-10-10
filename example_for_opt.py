#!/usr/bin/python
# !/usr/bin/env python


from other_algorithms import opt

import matplotlib.pyplot as plt
import time


def read_seq(file):
    sequence = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if len(line) == 0:
                continue
            data = line.split(',')
            sequence.append(data[0])
    f.close()
    return sequence


# Set parameters
seq = read_seq('datasets/gems_test.csv')
k = 128
number_of_box_kinds = 8
miss_cost = 1000
s = time.time()
opt_impact2, boxes, _ = opt(seq, k, number_of_box_kinds, miss_cost)
print('use', time.time() - s, 'secs')
print('opt impact =', opt_impact2)
print('opt box sequence is', boxes)

# draw memory profile
x = [0]
y = [0]
t = 0
for b in boxes:
    box_size = k / (2 ** b)
    x.append(t)
    y.append(box_size)
    t += box_size
    x.append(t)
    y.append(box_size)
x.append(t)
y.append(0)

plt.plot(x, y, linewidth=1, color="orange")
plt.xlabel('time (normalized)')
plt.ylabel('memory size')
plt.show()
