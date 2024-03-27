import copy
import time

from memory_profiler import memory_usage
from recbole.data import create_dataset, data_preparation
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from logging import getLogger
from recbole.model.general_recommender import *
from recbole.config import Config

from united_metric_of_recommender_systen.composite.composite_index import CompositeIndex
from united_metric_of_recommender_systen.composite.data_weight import DatasetWeight


class Runner:
    def __init__(self, alg, dataset_name: str = "ml_100k", is_logging: bool = False):
        self.is_logging = is_logging
        self.config = None
        self.dataset_name = dataset_name
        self.alg = alg
        self.model = None
        self.dataset = None
        self.trainer = None
        self.train_data, self.valid_data, self.test_data = None, None, None
        self.name_model = str(self.alg).split(".")[-1].split("'")[0]
        print(self.name_model)
        self.__pool()

    def __pool(self):
        self.__set_config({})
        if self.is_logging:
            self.__start_logger()
        self.__write_log(self.config)
        self.__create_dataset()
        self.__data_preparation()

    def __start_logger(self):
        init_logger(self.config)
        self.logger = getLogger()

    def __write_log(self, msg):
        if self.is_logging:
            self.logger.info(msg)

    def __create_dataset(self):
        self.dataset = create_dataset(self.config)
        self.__write_log(self.dataset)

    def __data_preparation(self):
        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.dataset)

    def __create_model(self):  # mutable
        self.model = self.alg(self.config, self.train_data.dataset).to(self.config['device'])

    def __create_trainer(self):  # mutable
        self.trainer = Trainer(self.config, self.model)

    def __set_config(self, rules):  # mutable
        parameter_dict = {
            'metrics': ['Recall', 'Precision', 'GAUC', 'MRR', 'NDCG', 'Hit', 'MAP', 'AveragePopularity', 'GiniIndex',
                        'ShannonEntropy'],
        }
        parameter_dict.update(rules)
        self.config = Config(model=self.name_model, dataset=self.dataset_name, config_dict=parameter_dict)

    def loop(self, rules):
        self.__set_config(rules)
        self.__create_model()
        self.__create_trainer()
        t_s = time.time()
        self.trainer.fit(self.train_data, self.valid_data)
        t_f = time.time()
        test_result = self.trainer.evaluate(self.test_data)
        t_e = time.time()

        def foo():
            a = copy.deepcopy(self.model)

        usage = memory_usage(foo)
        params = list(test_result.values())
        params.insert(0, t_e - t_f)
        params.insert(0, t_f - t_s)
        params.insert(0, usage[0])
        return CompositeIndex(params, DatasetWeight.ML100K).run()
