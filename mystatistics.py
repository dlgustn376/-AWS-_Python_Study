import math

def mean(num_list): # 평균
    return sum(num_list) / len(num_list)

def median(num_list): # 중앙값
    num_list.sort()
    if len(num_list) % 2 == 1:
        i = len(num_list) // 2
        return num_list[i]
    else:
        i = len(num_list) // 2
        return (num_list[i] + num_list[i - 1]) / 2

def dev(num_list): # 편차
    m = mean(num_list)
    return [x - m for x in num_list]

def var(num_list): # 분산
    n = len(num_list)
    d = dev(num_list)
    return sum([x * x for x in d]) / (n - 1)

def stdev(num_list): # 표준편차
    return math.sqrt(var(num_list))

def covar(list_1, list_2):  # 공분산
    n = len(list_1)
    list_1_dev = dev(list_1)
    list_2_dev = dev(list_2)
    return sum(x * y for x, y in zip(list_1_dev, list_2_dev)) / (n - 1)

def corel(list_1, list_2):  # 상관도 (-1 ~ 1)
    return covar(list_1, list_2) / (stdev(list_1) * stdev(list_2))