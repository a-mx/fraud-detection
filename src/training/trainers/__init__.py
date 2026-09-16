from src.training.trainers import baseline, mlp, xgb

TRAINERS = {
    "baseline": baseline.train,
    "xgb": xgb.train,
    "mlp": mlp.train,
}