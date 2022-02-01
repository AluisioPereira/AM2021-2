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
        sum_all = np.dot(p_x_w[k], Pw)
        for i in range(qtd_w):
            p_w_x[k, i] = (p_x_w[k, i] * Pw[i]) / sum_all

    return p_w_x


class ClassificaforBayesianoGaussiano(BaseEstimator, ClassifierMixin):
    def __init__(self, partition, Pw):
        self.partition = partition
        self.Pw = Pw

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self._fit_gaussian_bayesian_data(X)
        self.X_ = X
        self.y_ = y
        return self

    def _fit_gaussian_bayesian_data(self, X):
        n, d = X.shape
        qtd_w = len(self.partition)  # C
        # Média por dimensão para cada classe, 1 <= i <= 7
        # self.means.shape == (C, D)
        self.means = np.array([X[idxs].mean(axis=0) for idxs in self.partition])
        # sig2
        # self.var.shape == (C,)
        self.var = np.array(
            [
                ((X[idxs] - self.means[i]) ** 2).sum() / (d * n)
                for i, idxs in enumerate(self.partition)
            ]
        )
        # Ei
        self.cov_matrix = [np.zeros((d, d)) for _ in range(qtd_w)]

        for i in range(qtd_w):
            np.fill_diagonal(self.cov_matrix[i], self.var[i])

        return self

    def _calc_gaussian_density_prob(self, xk, cls):
        d = xk.shape[0]
        coef = np.power(2 * np.pi, -d / 2)
        inv_cov_matrix = np.linalg.inv(self.cov_matrix[cls])
        (sign, logdet) = np.linalg.slogdet(inv_cov_matrix)
        sqrt_det_inv_cov = np.sqrt(sign * np.exp(logdet))
        diff = xk - self.means[cls]
        exp_exp = np.dot((-1 / 2) * np.dot(diff.T, inv_cov_matrix), diff)
        exp_func = np.exp(exp_exp)

        return coef * sqrt_det_inv_cov * exp_func

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        desity_probs = np.empty((X.shape[0], len(self.classes_)))
        for k in range(desity_probs.shape[0]):
            for j in range(len(self.classes_)):
                desity_probs[k, j] = self._calc_gaussian_density_prob(X[k], j)

        post_probs = calc_prob_posteriori(desity_probs, self.Pw)

        return post_probs


class ClasificadorBayesianoGaussianoPorVotoMajoritario(BaseEstimator, ClassifierMixin):
    def __init__(self, partition, Pw):
        self.partition = partition
        self.Pw = Pw
        self.clfs = []

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
            clf = ClassificaforBayesianoGaussiano(self.partition, self.Pw)
            clf.fit(x, y)
            self.clfs.append(clf)

        return self

    def predict(self, X):
        assert len(X) == len(self.clfs)

        check_is_fitted(self)

        post_probs = [clf.predict_proba(x) for clf, x in zip(self.clfs, X)]

        return self.votacao(post_probs, Pw=self.Pw)

    def predict_with_proba(self, X):
        assert len(X) == len(self.clfs)

        check_is_fitted(self)

        return [clf.predict_proba(x).argmax(axis=1) for clf, x in zip(self.clfs, X)]

    def get_params(self, deep=True):
        return {"Pw": self.Pw, "y_true": self.partition}

    def votacao(self, matrizes, Pw):
        x_votes = np.array([m.argmax(axis=1) for m in matrizes]).transpose()
        y_pred = []
        for votes in x_votes:
            y_pred.append(Counter(votes).most_common()[0][0])
        return np.array(y_pred)


ClassificadorPorVoto = ClasificadorBayesianoGaussianoPorVotoMajoritario

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=VisibleDeprecationWarning, append=True)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning, append=True)


def testar(X_train, X_test, y_train, y_test):
    part_train = utils.get_particao_para_classificacao(y_train)
    priori_train = utils.calcular_probabilidade_priori(part_train)

    clf = ClassificadorPorVoto(part_train, priori_train)

    clf.fit(X_train, y_train)

    with open("../../reports/gaussian_classification_report.txt", "w") as f:
        report = classification_report(y_test, clf.predict(X_test))
        f.write(report)

    scoring = ("accuracy", "f1_weighted", "precision_weighted", "recall_weighted")
    scores = {}
    for sc_name in scoring:
        scores[sc_name] = metrics.get_scorer(sc_name)(clf, X_test, y_test)

    scores["error"] = 1 - scores["accuracy"]
    return report, scores


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

        part_train = utils.get_particao_para_classificacao(y_train)
        priori_train = utils.calcular_probabilidade_priori(part_train)
        clf = ClassificadorPorVoto(part_train, priori_train)

        clf.fit(X_train, y_train)

        for score_name, scorer in scoring.items():
            scores[score_name][i] = scorer(clf, X_test, y_test)

    scores_resume = ChainMap(
        *(
            {f"{name}_mean": sc.mean().round(3), f"{name}_std": sc.std()}
            for name, sc in scores.items()
        )
    )

    # for score_name, scs in scores.items():
    #     print(f"{score_name}:\n\tmean: {scs.mean()}\n\tstd: {scs.std()}")

    return dict(scores_resume), scores


def validar(dts, classes, seed=SEED):
    resume, all_scores = treinar(dts, classes, seed=seed)
    pd.DataFrame(all_scores).to_csv(
        "../../reports/gaussian_splits_scores.csv", index=False
    )
    best_result = resume
    pd.DataFrame([best_result]).to_csv(
        "../../reports/gaussian_best_result.csv", index=False
    )


if __name__ == "__main__":
    datasets = tuple(dataset.ALL_TEST.values())
    classes = dataset.CLASSES_TEST

    X_train, X_test, y_train, y_test = utils.datasets_train_test_split(
        datasets, classes, 0.20, SEED
    )

    validar(X_train, y_train)

    c_report, scores = testar(X_train, X_test, y_train, y_test)

    utils.exportar_intervalo_confianca_metricas(
        scores=scores,
        n=len(datasets[0]),
        arquivo="../../reports/gaussian_estimativa_pontual_intervalo.txt",
    )

    resume, all_scores = treinar(datasets, classes, seed=SEED)

    pd.DataFrame(all_scores).to_csv(
        "../../reports/gaussian_scores_for_friedman_test.csv", index=False
    )
