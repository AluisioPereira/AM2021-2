import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
import pickle

SEED = 42


def get_fmeasure_modificada(y, pertinencia):
    def precisao(pi, pk):
        return len(set(pi) & set(pk)) / len(pk)

    def cobertura(pi, pk):
        return len(set(pi) & set(pk)) / len(pi)

    def fm(pi, pk):
        try:
            return (
                2
                * (precisao(pi, pk) * cobertura(pi, pk))
                / (precisao(pi, pk) + cobertura(pi, pk))
            )
        except ZeroDivisionError:
            return 0

    y_class = get_particao_para_classificacao(y)
    y_cluster, _ = get_hard_patitions_clustering(pertinencia)

    C = len(y_class)
    N = sum(map(len, y_cluster))

    soma = 0
    for i in range(C):
        pi = y_class[i]
        bi = len(pi)
        soma += bi * max([fm(pi, qk) for qk in y_cluster])

    return soma / N


def calcular_probabilidade_priori(particao):
    tamanho = sum(map(len, particao))
    return np.array([len(membros) / tamanho for membros in particao])


def get_hard_patitions_clustering(pertinencia):
    """
        pertinencia: n.array de dimensões P x K
    """

    K = pertinencia.shape[1]

    # Obtendo o índice do grupo em que cada elemento possui maior valor de pertencimento
    pert = pertinencia.argsort(axis=1)[:, -1]
    membros = [list(np.where(pert == k)[0]) for k in range(K)]

    return membros, pert


def get_particao_para_classificacao(y):
    assert set(y) == set(range(max(y) + 1))
    y = np.array(y)
    part = [(y == i).nonzero()[0] for i in range(max(y) + 1)]
    return part


def datasets_train_test_split(datasets, classes, test_size, seed=SEED):
    assert datasets
    train, test = [], []
    class_train, class_test = [], []

    for dt in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dt, classes, test_size=test_size, random_state=seed
        )

        train.append(X_train)
        test.append(X_test)
        class_train.append(y_train)
        class_test.append(y_test)

    for i in range(len(datasets) - 1):
        assert tuple(class_train[i]) == tuple(class_train[i + 1])
        assert tuple(class_test[i]) == tuple(class_test[i + 1])

    return train, test, class_train[0], class_test[0]


def calcular_intervalo_confianca(p, n, z=1.96):
    diff = z * np.sqrt(p * (1 - p) / n)
    return round(p - diff, 4), round(p + diff, 4)


def exportar_intervalo_confianca_metricas(scores, n, arquivo, z=1.96):
    with open(arquivo, "w") as f:
        for sc_name, p in scores.items():
            intervalo = calcular_intervalo_confianca(p, n, z)
            f.write(f">> {sc_name}\n")
            f.write(f"> Estimativa pontual: {p}\n")
            f.write(f"> Intervalo de confiança: {intervalo}\n\n")


def ler_resultado(arquivo):
    with open(arquivo, "br") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import dataset

    y_true = dataset.CLASSES_TEST
    for name in dataset.ALL_TEST.keys():
        # if name not in ("rgb_train",):
        #     continue
        result = ler_resultado(f"../data/{name}.pickle")

        print(
            f"Medida-f {name}: ", get_fmeasure_modificada(y_true, result["pertinencia"])
        )
