import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import classification_report
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval


def my_cv(dict_cv, metrics, train=True):
    print("Cross Validation Scores:")
    if train:
        tt = ["train", "test"]
    else:
        tt = ["test"]
    for t in tt:
        print(t.capitalize())
        for metric in metrics:
            scoreprint = metric.replace("_", " ").capitalize()
            media = round(np.mean(dict_cv[t+"_"+metric]), 4)
            desvio = round(np.mean(dict_cv[t+"_"+metric]), 4)
            print(scoreprint + ":", media, "±", desvio)
    print()


class HyperOptCV:
    def __init__(self, x, y, model, scorer, kfold, regress=False, multiclass=False, smote=None, pca=None,
                 groups=None, scaler=None):
        self.x = x
        self.y = y
        self.model = model
        self.kfold = kfold
        self.groups = groups
        self.scorer = scorer
        self.regress = regress
        self.multiclass = multiclass
        self.pipe_method = Pipeline
        self.gpu = False
        self.best_params = None
        self.best_estimator = {}
        self.pca = pca

        if scaler is not None:
            self.scaler = scaler()
        else:
            self.scaler = None
            
        if smote is not None:
            from imblearn.pipeline import Pipeline as SmotePipeline
            self.smote = smote(random_state=42)
            self.pipe_method = SmotePipeline
        else:
            self.smote = None
            
    
        if ("XGB" in self.model.__name__) or ("LGBM" in self.model.__name__):
            self.space = {
                'n_estimators': hp.choice('n_estimators', np.arange(100, 1000, 100)),
                'learning_rate': hp.choice('learning_rate', np.arange(0.02, 0.25, 0.01)),
                'max_depth': hp.choice('max_depth', np.arange(5, 16, 1, dtype=int)),
                'min_child_weight': hp.choice('min_child_weight', np.arange(1, 7, 1, dtype=int)),
                'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.1, 0.8, 0.1)),
                'subsample': hp.choice('subsample', np.arange(0.1, 1, 0.1)),
                'gamma': hp.choice('gamma', [0, 1, 5]),
                "random_state": 42,
                'use_label_encoder': False,
                'objective': "binary:logistic",
                'eval_metric': 'logloss'
            }
            self.biblioteca_modelo = "XGB"
            if "LGBM" in self.model.__name__:
                self.space["objective"] = "binary"
                self.space.pop('use_label_encoder')
                self.space.pop('gamma')
                self.space.pop('eval_metric')
                self.biblioteca_modelo = "LGBM"
        else:
            self.space = dict()
            print("É preciso setar o espaço de hiperparâmetros para modelos do tipo", self.model.__name__)

        if self.regress:
            if self.biblioteca_modelo == "XGB":
                self.space["objective"] = "reg:squarederror"
                self.space.pop('eval_metric')
            elif self.biblioteca_modelo == "LGBM":
                self.space["objective"] = "regression"
            self.metrics = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2", "explained_variance"]
        else:
            if self.multiclass:
                if self.biblioteca_modelo == "XGB":
                    self.space.pop('eval_metric')
                    self.space["objective"] = "multi:softmax"
                elif self.biblioteca_modelo == "LGBM":
                    self.space["objective"] = "multiclass"
                self.metrics = ["accuracy", "balanced_accuracy", "f1_macro", 'f1_weighted']
            else:
                self.metrics = ["accuracy", "precision", "recall", "roc_auc", "balanced_accuracy", "f1"]

    def set_space(self, space):
        self.space = space

    def activate_gpu(self):
        self.gpu = True
        if self.biblioteca_modelo == "XGB":
            self.space['tree_method'] = 'gpu_hist'
            self.space['predictor'] = "gpu_predictor"
        elif self.biblioteca_modelo == "LGBM":
            self.space['device_type'] = 'gpu'
        else:
            print("Implementação de GPU não está configurada para modelo", self.model.__name__)

    def set_fixed_number_of_estimators(self, n_estimators):
        self.space["n_estimators"] = n_estimators

    def hyperparameter_tuning(self, space):
        model = self.model(**space)
        pipe = []
        if self.scaler is not None:
            pipe.append(('scaler', self.scaler))
        if self.smote is not None:
            pipe.append(('oversampler', self.smote))
        if self.pca is not None:
            pipe.append(('pca', self.pca))
        pipe.append(('estimator', model))
        self.pipeline = self.pipe_method(pipe)
        
        cv_score = cross_val_score(self.pipeline, self.x, self.y, groups=self.groups, cv=self.kfold, scoring=self.scorer)
        score = np.mean(cv_score)
        score_std = np.std(cv_score)
        if not self.silent:
            print(f"{self.scorer=}: " + str(score) + " ± " + str(round(score_std, 4)))
        return {'loss': -score, 'status': STATUS_OK, 'model': model}

    def get_optimized_params(self, epochs, silent=False):
        self.silent = silent
        trials = Trials()
        best = fmin(fn=self.hyperparameter_tuning,
            space=self.space,
            algo=tpe.suggest,
            max_evals=epochs,
            trials=trials,
            rstate=np.random.RandomState(42))
        self.best_params = space_eval(self.space, best)
        self.best_estimator = self.model(**self.best_params)
        return self.best_params

    def get_cv_metrics(self, target_names=None, metrics=None, return_train_score=False):
        assert self.best_params != None, 'É necessário obter os parâmetros antes. Rode hyper.get_optimized_params()'
        if metrics is None:
            metrics = self.metrics
        cv = cross_validate(self.pipeline, self.x, self.y, groups=self.groups, cv=self.kfold, scoring=metrics,
                            return_train_score=return_train_score)
        
        my_cv(cv, metrics)
        if self.multiclass:
            y_pred = cross_val_predict(self.pipeline, self.x, self.y, groups=self.groups, cv=self.kfold)
            print(classification_report(self.y, y_pred, target_names=target_names))
        return cv
    
    def get_cv_y_pred(self):
        y_pred = cross_val_predict(self.pipeline, self.x, self.y, groups=self.groups, cv=self.kfold)
        return y_pred
