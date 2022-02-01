import warnings
from collections import Counter, ChainMap

import numpy as np
import pandas as pd
from numpy import VisibleDeprecationWarning
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted
from tqdm import tqdm

from src import dataset
from src import utils
from src.utils import SEED


def calc_prob_posteriori(p_x_w, Pw):
    qtd_x, qtd_w = p_x_w.shape
    p_w_x = np.empty((qtd_x, qtd_w))

    for k in range(qtd_x):
        for i in range(qtd_w):
            sum_all = np.dot(p_x_w[k], Pw)
            p_w_x[k, i] = (p_x_w[k, i] * Pw[i]) / sum_all

    return p_w_x


class RegressaoLogistica(BaseEstimator, ClassifierMixin):
    def __init__(self, seed=SEED, **lr_args):
        self.clfs = {}
        self.lr_args = {"random_state": seed, **lr_args}

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self._fit(X, y)
        self._fitted = True
        return self

    def _fit(self, X, y):
        for c in self.classes_:
            yy = y.copy()
            yy[y == c] = 1
            yy[y != c] = 0
            self.clfs[c] = LogisticRegression(**self.lr_args).fit(X, yy)

    def predict(self, X):
        check_is_fitted(self)
        y_pred = []
        for x in X:
            for c, clf in self.clfs.items():
                pred = clf.predict([x])
                if pred == 1:
                    y_pred.append(c)
                    break
            else:
                y_pred.append(-1)

        return np.array(y_pred)


class ClasificadorRegressaoLogisticaPorVotoMajoritario(BaseEstimator, ClassifierMixin):
    def __init__(self, seed=SEED, **lr_args):
        self.clfs = []
        self.lr_args = {"random_state": seed, **lr_args}

    def fit(self, X, y):
        """

        :param X: Lista dos dts
        :param y: Rótulos
        :return: self (fitted)
        """
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        for x in X:
            clf = RegressaoLogistica(**self.lr_args)
            clf.fit(x, y)
            self.clfs.append(clf)

        return self

    def predict(self, X):
        assert len(X) == len(self.clfs)

        check_is_fitted(self)

        y_pred_all = [clf.predict(x) for clf, x in zip(self.clfs, X)]

        return self.votacao(y_pred_all)

    def get_params(self, deep=True):
        return {}

    def votacao(self, y_pred_all):
        x_votes = np.array(y_pred_all).transpose()
        y_pred = []
        for votes in x_votes:
            y_pred.append(Counter(votes).most_common()[0][0])

        return np.array(y_pred)


ClassificadorPorVoto = ClasificadorRegressaoLogisticaPorVotoMajoritario


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning, append=True)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning, append=True)


def treinar(dts, y, qtd_treinos=30, folds=10, seed=SEED):
    assert len(dts) == 3

    cv = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=qtd_treinos, random_state=seed
    )

    scoring = ("accuracy", "f1_weighted", "precision_weighted")
    scoring = {sc: metrics.get_scorer(sc) for sc in scoring}

    scores = {sc: np.empty(qtd_treinos * folds) for sc in scoring}
    split = enumerate(cv.split(dts[0], y))

    for i, (train_index, test_index) in tqdm(split, desc="Treinamento: "):
        X_train = [scale(dt[train_index]) for dt in dts]
        X_test = [scale(dt[test_index]) for dt in dts]
        y_train, y_test = y[train_index], y[test_index]

        clf = ClassificadorPorVoto(seed=seed)

        clf.fit(X_train, y_train)
        for score_name, scorer in scoring.items():
            scores[score_name][i] = scorer(clf, X_test, y_test)

    scores_resume = ChainMap(
        *(
            {f"{name}_mean": sc.mean().round(3), f"{name}_std": sc.std()}
            for name, sc in scores.items()
        )
    )

    for score_name, scs in scores.items():
        print(f"{score_name}:\n\tmean: {scs.mean()}\n\tstd: {scs.std()}")

    return dict(scores_resume), scores


def validar(dts, classes, seed=SEED):
    resume, all_scores = treinar(dts, classes, seed=seed)
    pd.DataFrame(all_scores).to_csv(
        "../../reports/logistic_regression_splits_scores.csv", index=False
    )


def testar(X_train, X_test, y_train, y_test, seed=SEED):

    clf = ClassificadorPorVoto(seed=seed)

    clf.fit(X_train, y_train)

    with open("../../reports/logistic_regression_classification_report.txt", "w") as f:
        report = classification_report(y_test, clf.predict(X_test))
        f.write(report)

    scoring = ("accuracy", "f1_weighted", "precision_weighted", "recall_weighted")

    scores = {}

    for sc_name in scoring:
        scores[sc_name] = metrics.get_scorer(sc_name)(clf, X_test, y_test)

    scores["error"] = 1 - scores["accuracy"]
    return report, scores


if __name__ == "__main__":
    datasets = tuple(dataset.ALL_TEST.values())
    classes = dataset.CLASSES_TEST

    X_train, X_test, y_train, y_test = utils.datasets_train_test_split(
        datasets, classes, 0.20, SEED
    )

    validar(datasets, classes)

    c_report, scores = testar(X_train, X_test, y_train, y_test)

    utils.exportar_intervalo_confianca_metricas(
        scores=scores,
        n=len(datasets[0]),
        arquivo="../../reports/logistic_regression_estimativa_pontual_intervalo.txt",
    )
    resume, all_scores = treinar(datasets, classes, seed=SEED)

    pd.DataFrame(all_scores).to_csv(
        "../../reports/logistic_regression_scores_for_friedman_test.csv", index=False
    )
