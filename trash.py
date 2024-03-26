from recbole.model.general_recommender import ItemKNN

a = ItemKNN

print(str(a).split(".")[-1].split("'")[0])