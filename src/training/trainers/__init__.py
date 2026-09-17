from src.training.trainers import baseline, mlp, xgb, randomforest

TRAINERS = {
    "baseline": baseline.train,
    "xgb": xgb.train,
    "mlp": mlp.train,
    "random_forest": randomforest.train
}