from recbole.data import Interaction, create_dataset, data_preparation
from recbole.evaluator import Evaluator
from recbole.model.general_recommender import SLIMElastic, ItemKNN
from recbole.config import Config
import numpy as np
from recbole.trainer import Trainer

# 1. Конфигурация и обучение первой модели
conf_dict = {
        'metrics': ['Recall', 'Precision', 'GAUC', 'MRR', 'NDCG', 'Hit', 'MAP', 'AveragePopularity',
                    'GiniIndex', 'ShannonEntropy'],
        'epochs':1
    }




config_1 = Config(model='SLIMElastic', dataset='ml-100k', config_dict=conf_dict)
# model_1, train_data, valid_data, test_data = run_recbole(model='LightGCN', dataset='ml-100k', config_dict=conf_dict)

dataset = create_dataset(config_1)
train_data, valid_data, test_data = data_preparation(config_1, dataset)
model_1 = SLIMElastic(config_1, train_data.dataset).to(config_1['device'])
trainer = Trainer(config_1, model_1)
trainer.fit(train_data, valid_data)

# 2. Конфигурация и обучение второй модели
config_2 = Config(model='ItemKNN', dataset='ml-100k', config_dict=conf_dict)
model_2 = ItemKNN(config_2, train_data.dataset).to(config_2['device'])
trainer = Trainer(config_2, model_2)
trainer.fit(train_data, valid_data)

# 3. Получение предсказаний базовых моделей
all_preds_1 = []

# Пройдитесь по всем батчам данных в FullSortEvalDataLoader
for batch_data in test_data:
    # Сделайте предсказание для каждого батча
    batch_preds = model_1.full_sort_predict(batch_data.to(config_1['device'])).detach().cpu().numpy()
    all_preds_1.append(batch_preds)

# Объединение предсказаний всех батчей в один массив
all_preds_1 = np.concatenate(all_preds_1, axis=0)


all_preds_2 = []
for batch_data in test_data:
    batch_preds = model_2.full_sort_predict(batch_data.to(config_1['device'])).detach().cpu().numpy()
    all_preds_2.append(batch_preds)

all_preds_2 = np.concatenate(all_preds_2, axis=0)

# Усреднение предсказаний двух моделей


# Соединение предсказаний в новый обучающий набор для мета-модели
X_train_meta = np.vstack((all_preds_1, all_preds_2)).T
y_train_meta = train_data['label'].detach().numpy()  # Реальные метки для обучения мета-модели


from sklearn.linear_model import LogisticRegression

# 4. Обучение мета-модели
meta_model = LogisticRegression()
meta_model.fit(X_train_meta, y_train_meta)


# 5. Получение предсказаний базовых моделей на тестовых данных
test_preds_1 = model_1.full_sort_predict(test_data).detach().numpy()
test_preds_2 = model_2.full_sort_predict(test_data).detach().numpy()

# Создание набора данных для тестирования мета-модели
X_test_meta = np.vstack((test_preds_1, test_preds_2)).T

# 6. Предсказание мета-модели
final_preds = meta_model.predict_proba(X_test_meta)[:, 1]

# Преобразование предсказаний в формат Interaction для расчета метрик
final_interaction = Interaction({'score': final_preds})

# 7. Расчет метрик
evaluator = Evaluator(config_1)  # Вы можете использовать конфигурацию одной из базовых моделей
metrics = evaluator.evaluate(final_interaction, test_data)

# 8. Печать результатов
for metric_name, metric_value in metrics.items():
    print(f'{metric_name}: {metric_value}')



