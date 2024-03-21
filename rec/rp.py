import time
from logging import getLogger

from memory_profiler import profile, memory_usage
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, LINE, NeuMF, ConvNCF, DMF, ItemKNN, NAIS, SpectralCF, GCMC, LightGCN, \
    DGCF, MultiVAE, MacridVAE, CDAE, RaCT, RecVAE, EASE, SLIMElastic, NCL, DiffRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import copy

from typing import List


class ComInd:
    def __init__(self, input_layer: List):
        self.input_layer = input_layer
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
        reverse_index = [0, 1, 2, 10]
        mins = [290, 0.4552, 0.1887, 0.0977, 0.0955, 0.6979, 0.2651, 0.1326, 0.4984, 0.0681, 173.5354, 0.8641, 0.009]
        min_max = [344.6, 112.1975, 1.0022, 0.1801, 0.121, 0.2214, 0.2709, 0.1919, 0.3298, 0.1342, 165.1135, 0.1234, 0.0474]

        for i in range(len(self.input_layer)):
            if i in reverse_index:
                self.before_normalize.append(1-((self.input_layer[i] - mins[i]) / min_max[i]))
                continue
            self.before_normalize.append((self.input_layer[i] - mins[i]) / min_max[i])


    def __layer_1(self):
        weight_1 = [0.280243293, 0.348150413, 0.371606293, 0.512949303, 0.487050697, 0.161179135, 0.196264416, 0.211645048, 0.221839635, 0.209071765, 0.291229633, 0.324222174,0.384548193]
        for i in range(len(self.before_normalize)):
            self.before_layer_1.append(self.before_normalize[i] * weight_1[i])

    def __conv_1(self):
        self.before_conv_1.append(self.before_layer_1[0] + self.before_layer_1[1] + self.before_layer_1[2])
        self.before_conv_1.append(self.before_layer_1[3] + self.before_layer_1[4])
        self.before_conv_1.append(self.before_layer_1[5] + self.before_layer_1[6] + self.before_layer_1[7] + self.before_layer_1[8] + self.before_layer_1[9])
        self.before_conv_1.append(self.before_layer_1[10] + self.before_layer_1[11] + self.before_layer_1[12])

    def __layer_2(self):
        weight_2 = [0.274425433, 0.303791675, 0.286439861, 0.135343031]
        for i in range(len(self.before_conv_1)):
            self.before_layer_2.append(self.before_conv_1[i] * weight_2[i])

    def __conv_2(self):
        self.score = sum(self.before_layer_2)


# K 1 - 1000 step 10
# SHRINK - 0 - 1 step 0.01
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    scores = []
    parameter_dict = {
        'metrics': ['Recall', 'Precision', 'GAUC', 'MRR', 'NDCG', 'Hit', 'MAP', 'AveragePopularity',
                    'GiniIndex', 'ShannonEntropy'],
        'epochs': 1,
        'k': 5,
        'shrink': 100 / 100,

    }
    config = Config(model='ItemKNN', dataset='Amazon_Software', config_dict=parameter_dict)
    dataset = create_dataset(config)


    train_data, valid_data, test_data = data_preparation(config, dataset)
    for k in range(1, 501, 10):
        temp = []
        for shrink in range(0, 100, 10):
            print(f"1")
            parameter_dict = {
                'metrics': ['Recall', 'Precision', 'GAUC', 'MRR', 'NDCG', 'Hit', 'MAP', 'AveragePopularity',
                            'GiniIndex', 'ShannonEntropy'],
                'epochs': 1,
                'k': k,
                'shrink': 0,

            }
            print(f"2")
            config = Config(model='ItemKNN', dataset='Amazon_Software', config_dict=parameter_dict)
            print(f"3")
            init_seed(config['seed'], config['reproducibility'])

            # init_logger(config)
            # logger = getLogger()

            # logger.info(config)


            model = ItemKNN(config, train_data.dataset).to(config['device'])
            print(f"4")
            trainer = Trainer(config, model)
            print(f"5")
            t_s = time.time()
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
            print(f"6")
            t_f = time.time()
            test_result = trainer.evaluate(test_data)
            print(f"7")
            t_e = time.time()


            def foo():
                a = copy.deepcopy(model)
            foo()
            print(f"8")
            usage = memory_usage(foo)
            params = list(test_result.values())
            params.insert(0, t_e - t_f)
            params.insert(0, t_f - t_s)
            params.insert(0, usage[0])

            temp.append(ComInd(params).run())
            print(f"9")
        scores.append(temp)
    f = open("cc.txt", "w")
    f.write(str(scores))
    print(scores)
