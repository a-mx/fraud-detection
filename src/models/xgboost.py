from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class XGBModel:
    def __init__(self, params, scale_pos_weight=None):
        self.model_params = {
            "n_estimators": params.n_estimators,
            "max_depth": params.max_depth,
            "learning_rate": params.learning_rate,
            "subsample": params.subsample,
            "colsample_bytree": params.colsample_bytree,
            "min_child_weight": params.min_child_weight,
            "reg_alpha": params.reg_alpha,
            "reg_lambda": params.reg_lambda,
            "n_jobs": params.n_jobs,
            "random_state": params.random_state,
            "eval_metric": "auc",
        }
        if scale_pos_weight is not None:
            self.model_params["scale_pos_weight"] = scale_pos_weight

        self.model = XGBClassifier(
            **self.model_params
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }