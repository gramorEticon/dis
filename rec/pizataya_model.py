from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
import torch
import torch.nn as nn


class PizModel(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(PizModel, self).__init__(config, dataset)
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.embedding_size = 64
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.predict_layer = nn.Linear(
            self.embedding_size, 1
        )
        self.sigmoid = nn.Sigmoid()
        self.loss = BPRLoss()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e = self.user_embedding(user)
        pos_item_e = self.item_embedding(pos_item)
        neg_item_e = self.item_embedding(neg_item)
        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1)

        loss = self.loss(pos_item_score, neg_item_score)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        scores = torch.mul(user_e, item_e).sum(dim=1)
        predict = self.sigmoid(self.forward(user, item))

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_e = self.user_embedding(user)  # [batch_size, embedding_size]
        all_item_e = self.item_embedding.weight  # [n_items, batch_size]

        scores = torch.matmul(user_e, all_item_e.transpose(0, 1))  # [batch_size, n_items]

        return scores