#!/usr/bin/python
# !/usr/bin/env python


def fun(x):
    y = x * 10
    arr = [i * x for i in range(3)]
    return y, arr


# y, arr = fun(10)
y = fun(10)
print(y)
# print(arr)


f = open("k.txt", "w")

f.writelines(str(y[1]))
f.close()

trace_names = ['astar', 'bwaves', 'bzip', 'cactusadm',
               'gems', 'lbm', 'leslie3d', 'libq', 'mcf',
               'milc', 'omnetpp', 'sphinx3', 'xalanc']

count = 0
for name in trace_names:
    f = open("datasets/"+name+'_test.csv', "r")
    count+=len(f.readlines())
    f.close()
print(count)
