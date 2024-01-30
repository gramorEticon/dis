import time


from rs_datasets import MovieLens

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.data.dataset_utils import DatasetLabelEncoder
from replay.metrics import HitRate, NDCG, Experiment, Novelty, Recall, Precision, Coverage, Mean, MAP, MRR, RocAuc, \
    Surprisal
from replay.models import *
from replay.utils.spark_utils import convert2spark
from replay.utils.session_handler import State
from replay.splitters import RatioSplitter



spark = State().session



ml_1m = MovieLens("10m")
K=10

# data preprocessing
interactions = convert2spark(ml_1m.ratings)

# data splitting
splitter = RatioSplitter(
    test_size=0.3,
    divide_column="user_id",
    query_column="user_id",
    item_column="item_id",
    timestamp_column="timestamp",
    drop_cold_items=True,
    drop_cold_users=True,
)
train, test = splitter.split(interactions)

# dataset creating
feature_schema = FeatureSchema(
    [
        FeatureInfo(
            column="user_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.QUERY_ID,
        ),
        FeatureInfo(
            column="item_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.ITEM_ID,
        ),
        FeatureInfo(
            column="rating",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.RATING,
        ),
        FeatureInfo(
            column="timestamp",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.TIMESTAMP,
        ),
    ]
)

train_dataset = Dataset(
    feature_schema=feature_schema,
    interactions=train,
)
test_dataset = Dataset(
    feature_schema=feature_schema,
    interactions=test,
)

# data encoding
encoder = DatasetLabelEncoder()
train_dataset = encoder.fit_transform(train_dataset)
test_dataset = encoder.transform(test_dataset)

#ALSWrap, ItemKNN, PopRec, RandomRec,
model_list = [ ALSWrap, ItemKNN, SLIM, Word2VecRec]

for elem in model_list:
    model = elem()
    t_s = time.time()
    model.fit(train_dataset)
    t_f = time.time()
    encoded_recs = model.predict(
        dataset=train_dataset,
        k=K,
        queries=test_dataset.query_ids,
        filter_seen_items=True,
    )
    t_e = time.time()

    recs = encoder.query_and_item_id_encoder.inverse_transform(encoded_recs)

    print(
        str(elem),
        "Time fit", t_f - t_s, "Time Predict", t_e - t_f,
        NDCG(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs, test_dataset.interactions),
        MAP(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs, test_dataset.interactions),
        HitRate(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs,
                                                                                          test_dataset.interactions),
        Coverage(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs,
                                                                                           test_dataset.interactions),
        MRR(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs, test_dataset.interactions),
        Novelty(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs,
                                                                                          test_dataset.interactions),
        Precision(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs,
                                                                                            test_dataset.interactions),
        Recall(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs,
                                                                                         test_dataset.interactions),
        RocAuc(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs,
                                                                                         test_dataset.interactions),
        Surprisal(K, query_column="user_id", item_column="item_id", rating_column="rating")(recs,
                                                                                            test_dataset.interactions),
    )


spark.stop()
