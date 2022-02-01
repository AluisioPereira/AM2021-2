import warnings
from collections import Counter, ChainMap

import numpy as np
import pandas as pd
from numpy import VisibleDeprecationWarning
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm import tqdm

import src.utils as utils
from src import dataset
from src.utils import SEED


def calc_prob_posteriori(p_x_w, Pw):
    qtd_x, qtd_w = p_x_w.shape
    p_w_x = np.empty((qtd_x, qtd_w))

    for k in range(qtd_x):
        for i in range(qtd_w):
            sum_all = np.dot(p_x_w[k], Pw)
            p_w_x[k, i] = (p_x_w[k, i] * Pw[i]) / sum_all

    return p_w_x


class ClassificaforBayesianoKnn(BaseEstimator, ClassifierMixin):
    def __init__(self, partition, Pw, k):
        self.partition = partition
        self.Pw = Pw
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self._fitted = True
        return self

    def _calc_knn_density_prob(self, X_dists):
        y_true = self.y_
        qtd_w = len(self.Pw)

        p_x_w = np.empty((qtd_w,))
        k_vizinhos = X_dists.argsort()[: self.k]

        w_vizinhos = y_true[k_vizinhos]
        for i in range(qtd_w):
            p_x_w[i] = (w_vizinhos == i).sum() / self.k

        return p_x_w

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_dists = euclidean_distances(X, self.X_)

        desity_probs = np.empty((X.shape[0], len(self.classes_)))
        for k in range(desity_probs.shape[0]):
            desity_probs[k] = self._calc_knn_density_prob(X_dists[k])

        post_probs = calc_prob_posteriori(desity_probs, self.Pw)

        return post_probs


class ClasificadorBayesianoKnnPorVotoMajoritario(BaseEstimator, ClassifierMixin):
    def __init__(self, partition, Pw, k):
        self.partition = partition
        self.Pw = Pw
        self.clfs = []
        self.k = k

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
            clf = ClassificaforBayesianoKnn(self.partition, self.Pw, self.k)
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
        return {"Pw": self.Pw, "y_true": self.partition, "k": self.k}

    def votacao(self, matrizes):
        x_votes = np.array([m.argmax(axis=1) for m in matrizes]).transpose()
        y_pred = []
        for votes in x_votes:
            y_pred.append(Counter(votes).most_common()[0][0])

        return np.array(y_pred)


ClassificadorPorVoto = ClasificadorBayesianoKnnPorVotoMajoritario


def testar(X_train, X_test, y_train, y_test, k):
    part_train = utils.get_particao_para_classificacao(y_train)
    priori_train = utils.calcular_probabilidade_priori(part_train)

    clf = ClassificadorPorVoto(part_train, priori_train, k=k)

    clf.fit(X_train, y_train)

    with open("../../reports/knn_classification_report.txt", "w") as f:
        report = classification_report(y_test, clf.predict(X_test))
        f.write(report)

    scoring = ("accuracy", "f1_weighted", "precision_weighted", "recall_weighted")
    scores = {}
    for sc_name in scoring:
        scores[sc_name] = metrics.get_scorer(sc_name)(clf, X_test, y_test)

    scores["error"] = 1 - scores["accuracy"]
    return report, scores


def treinar(dts, y, qtd_treinos=30, folds=10, k=5, seed=SEED):
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

        clf = ClassificadorPorVoto(part_train, priori_train, k=k)

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


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning, append=True)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning, append=True)


def validar(dts, classes, k_range=(1, 7), seed=SEED):
    results = []
    report = []
    for k in range(*k_range):
        print("K:", k)
        resume, all_scores = treinar(dts, classes, k=k, seed=seed)
        report.append(all_scores)
        resume["k"] = k
        all_scores["k"] = [k] * 300
        results.append(resume)

    report = pd.concat(map(pd.DataFrame, report))
    pd.DataFrame(report).to_csv("../../reports/knn_splits_scores.csv", index=False)
    best_result = max(results, key=lambda r: r["f1_weighted_mean"])
    pd.DataFrame([best_result]).to_csv("../../reports/knn_best_result.csv", index=False)

    return best_result


if __name__ == "__main__":
    datasets = tuple(dataset.ALL_TEST.values())
    classes = dataset.CLASSES_TEST

    X_train, X_test, y_train, y_test = utils.datasets_train_test_split(
        datasets, classes, 0.20, SEED
    )

    best_result = validar(X_train, y_train)

    c_report, scores = testar(X_train, X_test, y_train, y_test, k=best_result["k"])

    utils.exportar_intervalo_confianca_metricas(
        scores=scores,
        n=len(datasets[0]),
        arquivo="../../reports/knn_estimativa_pontual_intervalo.txt",
    )

    resume, all_scores = treinar(datasets, classes, k=best_result["k"], seed=SEED)

    pd.DataFrame(all_scores).to_csv(
        "../../reports/knn_scores_for_friedman_test.csv", index=False
    )
