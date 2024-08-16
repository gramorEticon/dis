
weight_first = [0.25, 0.4, 0.15, 0.2]


def local_decompose(rate, weight):
    loc = []
    for i in range(0, len(weight)):
        loc.append(rate * weight[i])
        print(loc[i])
    return loc

def recovery_rate(rate, weigth):
    for i in range(0, len(rate)):
        rate[i] = rate[i]/ weigth[i]
    return rate

lens = []
for r in range(0, 100):
    rate = r / 100
    arr = []
    for l in range(0,100, 2):
        for k in range(0,100, 2):
            for i in range(0, 100, 2):
                for j in range(0, 100, 2):
                    i_ = i / 100
                    j_ = j / 100
                    k_ = k / 100
                    l_ = l /100
                    if abs((i_*weight_first[0] + j_*weight_first[1]+ k_*weight_first[2]+ l_*weight_first[3])-rate) < 0.000001:
                        arr.append([rate])
    lens.append(len(arr))
    print(len(arr), rate)
print(lens)

import matplotlib.pyplot as plt

plt.plot(lens)
plt.show()
# z*x + z2 * x2