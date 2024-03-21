from typing import List


class CompositeIndex:
    def __init__(self, input_layer: List, weight):
        self.input_layer = input_layer
        self.weight = weight
        self.before_normalize = []
        self.before_layer_1 = []
        self.before_conv_1 = []
        self.before_layer_2 = []
        self.score = None

    def run(self):
        self.__normalize_layer()
        self.__layer_1()
        self.__conv_1()
        self.__layer_2()
        self.__conv_2()
        return self.score

    def __normalize_layer(self):
        for i in range(len(self.input_layer)):
            if i in self.weight.reverse_index:
                self.before_normalize.append(1 - ((self.input_layer[i] - self.weight.mins[i]) / self.weight.min_max[i]))
                continue
            self.before_normalize.append((self.input_layer[i] - self.weight.mins[i]) / self.weight.min_max[i])

    def __layer_1(self):
        for i in range(len(self.before_normalize)):
            self.before_layer_1.append(self.before_normalize[i] * self.weight.weight_1[i])

    def __conv_1(self):
        self.before_conv_1.append(self.before_layer_1[0] + self.before_layer_1[1] + self.before_layer_1[2])
        self.before_conv_1.append(self.before_layer_1[3] + self.before_layer_1[4])
        self.before_conv_1.append(self.before_layer_1[5] + self.before_layer_1[6] + self.before_layer_1[7] + self.before_layer_1[8] + self.before_layer_1[9])
        self.before_conv_1.append(self.before_layer_1[10] + self.before_layer_1[11] + self.before_layer_1[12])

    def __layer_2(self):
        for i in range(len(self.before_conv_1)):
            self.before_layer_2.append(self.before_conv_1[i] * self.weight.weight_2[i])

    def __conv_2(self):
        self.score = sum(self.before_layer_2)
