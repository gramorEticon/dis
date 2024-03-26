import math
import random
from typing import List

from recbole.model.general_recommender import ItemKNN, LINE

from united_metric_of_recommender_systen.runner.runner import Runner


class QuantOptimizer:
    def __init__(self, count_dots, odz, count_iteration=10):
        self.dots: List = []
        self.count_dots: int = count_dots if count_dots % 2 == 0 else count_dots + 1
        self.odz: List = odz
        self.count_iteration: int = count_iteration
        self.rec = Runner(LINE, "ml-100k", is_logging=False)
        self.cout = 0
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
            self.dots.append(temp)


    def __f(self, x):
        #return math.cos(x[0] / 25) + math.sin(x[1] / 25) + 0.03 * x[1] + 0.03 * x[0]
        self.cout += 1
        print(self.cout)
        return self.rec.loop(x[0], x[1], x[2]/100)
       # return self.rec.loop(100, 2, 1)

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
            print(self.dots)
            random.shuffle(self.dots)

    def __find_min(self):
        mx = self.dots[0][-1]
        bst = None
        for elem in self.dots:
            if elem[-1] >= mx:
                mx = elem[-1]
                bst = elem[0:-1]
        print(bst, mx)

if __name__ == "__main__":
    QuantOptimizer(30, [[32, 512, True], [1, 2, True], [0, 100, False]],)
