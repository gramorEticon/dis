from optuna.samplers import *
from recbole.model.general_recommender import SLIMElastic, CDAE, ItemKNN

from united_metric_of_recommender_systen.quant_optimizer.quant_optimizer import QuantOptimizer
import warnings
import optuna

from united_metric_of_recommender_systen.runner.runner_opt import RunnerOpt

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # QuantOptimizer(30, [[32, 512, True], [1, 2, True], [0, 100, False], [1,21, True]],)
    r = RunnerOpt(ItemKNN, "ml-100k", is_logging=False)


    # LINE
    # def objective(trial):
    #     epochs = trial.suggest_int("epochs", 0, 100)
    #     embedding_size = trial.suggest_int("embedding_size", 64, 1024)
    #     order = trial.suggest_int("order", 1, 2)
    #     second_order_loss_weight = trial.suggest_float("second_order_loss_weight", 0.001, 1)
    #     rule = {
    #         'epochs': epochs,
    #         'embedding_size': embedding_size,
    #         'order': order,
    #         'second_order_loss_weight': second_order_loss_weight
    #     }
    #     return r.loop(rule)

    # SLIM
    # def objective(trial):
    #     alpha = trial.suggest_float("alpha", 0.01, 0.99)
    #     l1_ratio = trial.suggest_float("l1_ratio", 0.001, 0.5)
    #     positive_only = trial.suggest_categorical("positive_only",[True, False])
    #     hide_item = trial.suggest_categorical("hide_item",[True, False])
    #     rule = {
    #         'alpha': alpha,
    #         'l1_ratio': l1_ratio,
    #         'positive_only': positive_only,
    #         'hide_item': hide_item,
    #         'epochs': 1
    #     }
    #     return r.loop(rule)


    # CDAE
    # def objective(trial):
    #     loss_type = trial.suggest_categorical("loss_type", ["BCE", "MSE"])
    #     hid_activation = trial.suggest_categorical("hid_activation", ["sigmoid", "relu", "tanh"])
    #     out_activation = trial.suggest_categorical("out_activation", ["sigmoid", "relu", "tanh"])
    #     corruption_ratio = trial.suggest_float("corruption_ratio", 0.01, 0.99)
    #     embedding_size = trial.suggest_int("embedding_size", 32, 1024)
    #     reg_weight_1 = trial.suggest_float("reg_weight_1", 0.001, 0.5)
    #     reg_weight_2 = trial.suggest_float("reg_weight_2", 0.001, 0.5)
    #
    #
    #     rule = {
    #         'loss_type': loss_type,
    #         'hid_activation': hid_activation,
    #         'out_activation': out_activation,
    #         'corruption_ratio': corruption_ratio,
    #         'embedding_size': embedding_size,
    #         'reg_weight_1': reg_weight_1,
    #         'reg_weight_2': reg_weight_2,
    #         'epochs': 30
    #     }
    #     return r.loop(rule)

    def objective(trial):
        k = trial.suggest_int("k", 10, 300)
        shrink = trial.suggest_float("shrink", 0.001, 0.5)

        rule = {
            'k': k,
            'shrink': shrink,
            'epochs': 1
        }
        return r.loop(rule)

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="ItemKNN-11",
        direction="maximize",
    )
    study.optimize(objective, n_trials=200)
    print(f"Best value: {study.best_value} (params: {study.best_params})")




