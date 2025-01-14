import math
import random
import time
from typing import List

from recbole.model.general_recommender import ItemKNN, LINE

from united_metric_of_recommender_systen.runner.runner import Runner


class QuantOptimizer:
    def __init__(self, count_dots, odz, count_iteration=10):
        self.dots: List = []
        self.count_dots: int = count_dots if count_dots % 2 == 0 else count_dots + 1
        self.odz: List = odz
        self.count_iteration: int = count_iteration
        self.rec = Runner(ItemKNN, "ml-100k", is_logging=False)
        self.count = 0
        self.range = count_dots + (count_iteration * (count_dots/2))
        self.t = None
        self.__pool()

    def __pool(self) -> None:
        self.__create_dots()
        self.__interation()
        self.__find_min()

    def __create_dots(self) -> None:
        for i in range(0, self.count_dots):
            temp = []
            for j in range(0, len(self.odz)):
                temp.append(random.randint(self.odz[j][0], self.odz[j][1]))
            temp.append(self.__f(temp))
            print(temp[-1])
            self.dots.append(temp)

    def __f(self, x):
        if self.t is None:
            self.t = time.time()
        else:
            if self.count == 1:
                temp = ((time.time() - self.t) * self.range) * 1.1
                print("Примерно осталось:",temp//60, "мин.", temp % 60, "сек.")
        self.count += 1

        rules = {
            'epochs': 1,
            'k': x[0],
            'shrink': x[1]/1000,
        }
        return self.rec.loop(rules)

    def __interation(self):
        for _ in range(0, self.count_iteration):
            new_dots = []
            for i in range(0, len(self.dots), 2):
                a = self.dots[i]
                b = self.dots[i + 1]
                c = []
                for j in range(0, len(a) - 1):  # Так как последнее это значение
                    if self.odz[j][-1]:
                        c.append(int((a[j] + b[j]) / 2))
                    else:
                        c.append((a[j] + b[j]) / 2)
                c.append(self.__f(c))
                if c[-1] >= a[-1]:
                    a = c
                else:
                    if c[-1] >= b[-1]:
                        b = c
                new_dots.append(a)
                new_dots.append(b)
            self.dots = new_dots
            random.shuffle(self.dots)

    def __find_min(self):
        mx = self.dots[0][-1]
        bst = None
        for elem in self.dots:
            if elem[-1] >= mx:
                mx = elem[-1]
                bst = elem[0:-1]
        print(bst, mx)
