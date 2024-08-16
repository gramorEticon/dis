import warnings

from united_metric_of_recommender_systen.quant_optimizer.quant_optimizer import QuantOptimizer

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    QuantOptimizer(30, [[10, 300, True], [1, 500, True]],)