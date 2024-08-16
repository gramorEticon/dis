import math
from enum import Enum
from typing import List


class NormMethod(Enum):
    DEFAULT = 0
    SIGMOID = 1
    SQRT_HS = 2
    SIGMOID_HS = 3


class Methods:
    @staticmethod
    def minmax(a):
        lst = []
        mn=min(a)
        mx = max(a)
        if mx-mn == 0:
            return [1 / len(a)] * len(a)
        for el in a:
            print(el)
            lst.append((el-mn)/(mx-mn)*100)
        return Methods.norm_form(lst)

    @staticmethod
    def sigmoid(a):
        lst = []
        for el in a:
            lst.append(101.35/(1+150*math.e**(-0.1*el))-0.671)
        print(lst)
        return Methods.norm_form(lst)

    @staticmethod
    def squirt_hs(a):
        lst=[]
        for el in a:
            lst.append(math.sqrt(100*el))
        return Methods.norm_form(lst)

    @staticmethod
    def sigmoid_hs(a):
        lst = []
        for el in a:
            lst.append(150 / (1 + 2 * math.e ** (-0.1 * el)) - 50)
        print(lst)
        return Methods.norm_form(lst)

    @staticmethod
    def default(a):
        return Methods.norm_form(a)

    @staticmethod
    def norm_form(a):
        lst = []
        s = sum(a)
        if s == 0:
            return [1/len(a)]*len(a)
        for el in a:
            lst.append(el/s)
        return lst
