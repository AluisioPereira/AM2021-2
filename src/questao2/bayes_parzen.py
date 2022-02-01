import warnings
from collections import Counter, ChainMap

import numpy as np
import pandas as pd
from numpy import VisibleDeprecationWarning
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm import tqdm

from src import utils, dataset
from src.utils import SEED


def calc_prob_posteriori(p_x_w, Pw):
    qtd_x, qtd_w = p_x_w.shape
    p_w_x = np.empty((qtd_x, qtd_w))

    for k in range(qtd_x):
        for i in range(qtd_w):
            sum_all = np.dot(p_x_w[k], Pw)
            p_w_x[k, i] = (p_x_w[k, i] * Pw[i]) / sum_all

    return p_w_x


class ClassificaforBayesianoParzen(BaseEstimator, ClassifierMixin):
    def __init__(self, partition, Pw, h):
        self.partition = partition
        self.Pw = Pw
        self.h = h

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self._fitted = True
        return self

    def _calc_parzen_density_prob(self, X, k, i):
        y_true = self.y_
        qtd_x, qtd_w = X.shape

        # p_x_w = np.empty((qtd_x, qtd_w))
        dims = X.shape[1]

        x_view = self.X_[y_true == i, :]
        n = x_view.shape[0]
        diff = (X[k] - x_view) / self.h
        gaussian_kernel = np.exp(-(diff ** 2) / 2) / np.sqrt(2 * np.pi)
        prod_dims = gaussian_kernel.prod(axis=1)
        return prod_dims.sum() / (n * self.h ** dims)

        # for i in range(qtd_w):
        #     x_view = self.X_[y_true == i, :]
        #     n = x_view.shape[0]
        #     for k in range(qtd_x):
        #         diff = (X[k] - x_view) / self.h
        #         gaussian_kernel = np.exp(-(diff ** 2) / 2) / np.sqrt(2 * np.pi)
        #         prod_dims = gaussian_kernel.prod(axis=1)
        #         p_x_w[k, i] = prod_dims.sum() / (n * self.h ** dims)
        #
        # return p_x_w

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        desity_probs = np.empty((X.shape[0], len(self.classes_)))
        # desity_probs = self._calc_parzen_density_prob(X)
        for k in range(desity_probs.shape[0]):
            for j in range(len(self.classes_)):
                desity_probs[k, j] = self._calc_parzen_density_prob(X, k, j)

        post_probs = calc_prob_posteriori(desity_probs, self.Pw)

        return post_probs


class ClasificadorBayesianoParzenPorVotoMajoritario(BaseEstimator, ClassifierMixin):
    def __init__(self, partition, Pw, h):
        self.partition = partition
        self.Pw = Pw
        self.clfs = []
        self.h = h

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
            clf = ClassificaforBayesianoParzen(self.partition, self.Pw, self.h)
            clf.fit(x, y)
            self.clfs.append(clf)

        return self

    def predict(self, X):
        assert len(X) == len(self.clfs)

        check_is_fitted(self)

        post_probs = [clf.predict_proba(x) for clf, x in zip(self.clfs, X)]

        return self.votacao(post_probs)

    def predict_with_proba(self, X, y_true):
        assert len(X) == len(self.clfs)

        check_is_fitted(self)

        return [
            clf.predict_proba(x, y_true).argmax(axis=1) for clf, x in zip(self.clfs, X)
        ]

    def get_params(self, deep=True):
        return {"Pw": self.Pw, "y_true": self.partition, "h": self.h}

    def votacao(self, matrizes):
        x_votes = np.array([m.argmax(axis=1) for m in matrizes]).transpose()
        y_pred = []
        for votes in x_votes:
            y_pred.append(Counter(votes).most_common()[0][0])

        return np.array(y_pred)


ClassificadorPorVoto = ClasificadorBayesianoParzenPorVotoMajoritario


def treinar(dts, y, qtd_treinos=30, folds=10, h=2, seed=SEED):
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
        part_train = utils.get_particao_para_classificacao(y_train)
        priori_train = utils.calcular_probabilidade_priori(part_train)

        clf = ClassificadorPorVoto(part_train, priori_train, h=h)

        clf.fit(X_train, y_train)
        for score_name, scorer in scoring.items():
            scores[score_name][i] = scorer(clf, X_test, y_test)

    scores_resume = ChainMap(
        *(
            {f"{name}_mean": sc.mean().round(3), f"{name}_std": sc.std()}
            for name, sc in scores.items()
        )
    )

    return dict(scores_resume), scores


def validar(dts, classes, h_range=(1, 7), seed=SEED):
    results = []
    report = []
    for h in range(*h_range):
        print("H:", h)
        resume, all_scores = treinar(dts, classes, h=h, seed=seed)
        report.append(all_scores)
        resume["h"] = h
        all_scores["h"] = [h] * 300
        results.append(resume)

    report = pd.concat(map(pd.DataFrame, report))
    pd.DataFrame(report).to_csv("../../reports/parzen_splits_scores.csv", index=False)
    best_result = max(results, key=lambda r: r["f1_weighted_mean"])
    pd.DataFrame([best_result]).to_csv(
        "../../reports/parzen_best_result.csv", index=False
    )

    return best_result


def testar(X_train, X_test, y_train, y_test, h):
    part_train = utils.get_particao_para_classificacao(y_train)
    priori_train = utils.calcular_probabilidade_priori(part_train)

    clf = ClassificadorPorVoto(part_train, priori_train, h=h)

    clf.fit(X_train, y_train)

    with open("../../reports/parzen_classification_report.txt", "w") as f:
        report = classification_report(y_test, clf.predict(X_test))
        f.write(report)

    scoring = ("accuracy", "f1_weighted", "precision_weighted", "recall_weighted")

    scores = {}
    for sc_name in scoring:
        scores[sc_name] = metrics.get_scorer(sc_name)(clf, X_test, y_test)

    scores["error"] = 1 - scores["accuracy"]
    return report, scores


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning, append=True)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning, append=True)

if __name__ == "__main__":
    datasets = tuple(dataset.ALL_TEST.values())
    classes = dataset.CLASSES_TEST

    X_train, X_test, y_train, y_test = utils.datasets_train_test_split(
        datasets, classes, 0.20, SEED
    )

    best_result = validar(X_train, y_train)

    c_report, scores = testar(X_train, X_test, y_train, y_test, h=best_result["h"])

    utils.exportar_intervalo_confianca_metricas(
        scores=scores,
        n=len(datasets[0]),
        arquivo="../../reports/parzen_estimativa_pontual_intervalo.txt",
    )

    resume, all_scores = treinar(datasets, classes, h=best_result["h"], seed=SEED)

    pd.DataFrame(all_scores).to_csv(
        "../../reports/parzen_scores_for_friedman_test.csv", index=False
    )
