from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from torchviz import make_dot

from rec.pizataya_model import PizModel


config = Config(model=PizModel, dataset='ml-100k')
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

model = PizModel(config, train_data)
y = model(train_data)
make_dot(y.mean(), params=dict(model.named_parameters()))