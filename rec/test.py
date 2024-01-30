import time
from logging import getLogger

from memory_profiler import profile
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, LINE, NeuMF, ConvNCF, DMF, ItemKNN, NAIS, SpectralCF, GCMC, LightGCN, \
    DGCF, MultiVAE
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import copy

if __name__ == '__main__':
    parameter_dict = {
        'metrics': ['Recall', 'Precision', 'GAUC', 'MRR', 'NDCG', 'Hit', 'MAP',  'AveragePopularity', 'GiniIndex', 'ShannonEntropy']
    }
    # configurations initialization
    config = Config(model='MultiVAE', dataset='ml-100k', config_dict=parameter_dict)


    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = MultiVAE(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)


    t_s = time.time()
    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    t_f = time.time()
    test_result = trainer.evaluate(test_data)
    t_e = time.time()
    print("Fit -> ", t_f-t_s, "Predict -> ", t_e-t_f)

    @profile
    def foo():
        a = copy.deepcopy(model)
    foo()

    print(test_result)