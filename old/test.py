import os
import time
from logging import getLogger

from memory_profiler import memory_usage
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import ItemKNN, LINE
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import copy

from united_metric_of_recommender_systen.composite.composite_index import CompositeIndex
from united_metric_of_recommender_systen.composite.data_weight import DatasetWeight

# K 1 - 1000 step 10
# SHRINK - 0 - 1 step 0.01
#[349, 2, 48.5, 17] 0.5966847085011683

if __name__ == '__main__':
    scores = []

    parameter_dict = {
        'metrics': ['Recall', 'Precision', 'GAUC', 'MRR', 'NDCG', 'Hit', 'MAP', 'AveragePopularity',
                    'GiniIndex', 'ShannonEntropy'],
        'epochs': 25,
        'embedding_size': 349,
        'order': 2,
        'second_order_loss_weight': 0.485

    }
    config = Config(model='LINE', dataset='ml-100k', config_dict=parameter_dict)
    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = LINE(config, train_data.dataset).to(config['device'])

    trainer = Trainer(config, model)

    t_s = time.time()
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    t_f = time.time()
    test_result = trainer.evaluate(test_data)
    t_e = time.time()

    def foo():
        a = copy.deepcopy(model)
    foo()

    os.remove(trainer.saved_model_file)
    usage = memory_usage(foo)
    params = list(test_result.values())
    params.insert(0, t_e - t_f)
    params.insert(0, t_f - t_s)
    params.insert(0, usage[0])
    print(params)

    print(CompositeIndex(params, DatasetWeight.ML100K).run())

