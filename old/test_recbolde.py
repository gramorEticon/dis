from logging import getLogger

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.context_aware_recommender import DeepFM
from recbole.model.general_recommender import BPR, NeuMF
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model, get_trainer


from torchviz import make_dot

from pizataya_model import PizModel

if __name__ == '__main__':

    # configurations initialization
    config = Config(model=NeuMF, dataset='ml-100k')
    print(config["device"])

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
    model = NeuMF(config, train_data.dataset).to(config['device'])
    logger.info(model)


    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)
    model.forward(1,2)





