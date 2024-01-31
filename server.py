import time
from logging import getLogger

from memory_profiler import profile
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR, LINE, NeuMF, ConvNCF, DMF, ItemKNN, NAIS, SpectralCF, GCMC, LightGCN, \
    DGCF, MultiVAE, MacridVAE, CDAE, RaCT, RecVAE, EASE, SLIMElastic, NCL, DiffRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import copy
str_mdl = ["LightGCN", "MultiVAE", "CDAE", "RaCT", "SLIMElastic", "ItemKNN", "DiffRec"]
smdl = [LightGCN, MultiVAE, CDAE, RaCT, SLIMElastic, ItemKNN, DiffRec]

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    for i in range(0, len(smdl)):

        parameter_dict = {
            'metrics': ['Recall', 'Precision', 'GAUC', 'MRR', 'NDCG', 'Hit', 'MAP',  'AveragePopularity', 'GiniIndex', 'ShannonEntropy'],
        }
        config = Config(model=str_mdl[i], dataset='ml-100k', config_dict=parameter_dict)
        init_seed(config['seed'], config['reproducibility'])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = smdl[i](config, train_data.dataset).to(config['device'])
        trainer = Trainer(config, model)
        t_s = time.time()
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
        t_f = time.time()
        test_result = trainer.evaluate(test_data)
        t_e = time.time()
        print("===============", str_mdl[i], "===============")
        print("Fit -> ", t_f-t_s, "Predict -> ", t_e-t_f)
        @profile
        def foo():
            a = copy.deepcopy(model)
        foo()
        print(test_result)
        print("")

