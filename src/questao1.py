"""
Implementações padrão das fórmulas requeridas para responder à questão 1,
sem preocupação com performance.

Para executar o treinamento conforme a questão pede e exportar os resultados para a pasta ../data,
execute esse script diretamente.

"""
import pickle
import sys
from operator import itemgetter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import minmax_scale, scale, robust_scale

from src.utils import SEED

C = 7  # Quantidade de clusters

T = 150  # Quantidade máxima de iterações
eps = 1e-10  # Tolerância

TU = 0.3  # Valor chutado
TV = 0.3  # Valor chutado
TOTAL_TREINAMENTOS = 50


def funcao_objetivo(
    dataset, matriz_pertinencia, matriz_relevancia, prototipos, Tu=TU, Tv=TV
):

    """
        Função objetivo (JEFCM-LS1) conforme definda na Eq (12), pg. 4

    :param dataset: Dataset a ser considerado
    :param matriz_pertinencia:
    :param matriz_relevancia:
    :param prototipos:
    :param Tu: Grau de fuzzificação
    :param Tv: Relevância das variáveis
    :return: (float) O valor da função objetivo
    """

    x = dataset
    u = matriz_pertinencia
    g = prototipos
    v = matriz_relevancia

    P, V = x.shape

    parcela1 = 0
    for k in range(C):
        for i in range(P):
            soma_aux = np.sum(v[k] * np.abs(x[i] - g[k]))
            parcela1 += u[i, k] * soma_aux

    u_log = np.log(u)
    u_log = np.nan_to_num(u_log, neginf=u_log.max())
    parcela2 = Tu * np.sum(u * u_log)
    v_log = np.log(v)
    v_log = np.nan_to_num(v_log, neginf=v_log.max())
    parcela3 = Tv * np.sum(v * v_log)

    return parcela1 + parcela2 + parcela3


def calcular_prototipos(dataset, matriz_pertinencia):
    """
        Função não finalizada

    :param dataset:
    :param matriz_pertinencia:
    :return: np.array
    """

    u = matriz_pertinencia
    x = dataset
    P, V = dataset.shape

    g = np.empty((C, V), dtype=float)

    for k in range(C):
        for j in range(V):
            yi = u[:, k] * x[:, j]
            zi = u[:, k]

            razao = yi / zi
            argsort = razao.argsort()
            new_yi, new_zi = yi[argsort], zi[argsort]
            soma = new_zi.sum()

            atual = seguinte = 0

            for r in range(P - 1):
                atual = -soma + 2 * np.abs(new_zi[: r + 1]).sum()
                seguinte = -soma + 2 * np.abs(new_zi[: r + 2]).sum()

                if atual < 0 < seguinte:
                    g[k, j] = new_yi[r] / new_zi[r]
                    break

            if atual == seguinte == 0:
                g[k, j] = (new_yi[r] / new_zi[r] + new_yi[r + 1] / new_zi[r + 1]) / 2

    return g


def calcular_relevancia(dataset, matriz_pertinencia, prototipos, Tv=TV):

    """
        Calcula o grau de relevância conforme equação 26
    :param dataset:
    :param matriz_pertinencia:
    :param prototipos:
    :return: np.array
        Matriz de relevância calculada
    """

    x = dataset
    u = matriz_pertinencia
    g = prototipos

    V = x.shape[1]
    v = np.empty((C, V), dtype=float)

    for k in range(C):
        soma_total = np.sum(
            [
                np.exp(-(u[:, k] * np.abs(x[:, w] - g[k, w])).sum() / Tv)
                for w in range(V)
            ]
        )
        for j in range(V):
            numerador = np.exp(-(u[:, k] * np.abs(x[:, j] - g[k, j])).sum() / Tv)
            v[k, j] = numerador / soma_total

    return v


def calcular_pertinencia(dataset, matriz_relevancia, prototipos, Tu=TU):

    """
        Função responsável pelo cálculo do grau de pertinência

    :param dataset:
    :param matriz_relevancia:
    :param prototipos:
    :param Tu:
    :return: np.array

    """
    x = dataset
    v = matriz_relevancia
    g = prototipos

    P, V = x.shape

    u = np.empty((P, C), dtype=float)

    for i in range(P):
        soma_total = np.sum(
            [np.exp(-(v[w] * np.abs(x[i] - g[w])).sum() / Tu) for w in range(C)]
        )
        for k in range(C):
            u[i, k] = np.exp(-(v[k] * np.abs(x[i] - g[k])).sum() / Tu) / soma_total

    return u


def executar_clusterizacao(dataset, seed=SEED, T=T, Tu=TU, Tv=TV, eps=eps):

    """
        Função responsável por executar um único treinamento com o test_dataset

    :param dataset: Dataset desejado
    :param seed: Semente para garantir reprodutibilidade
    :param T: Quantidade máxima de iterações
    :param Tu: Grau de fuzzificação
    :param Tv: Relevância das variáveis
    :param eps: Limitar de tolerência

    :return: Dicionário contendo prototipos, relevancia, pertinencia o menor e último custo calculado
    """

    np.random.seed(seed)
    P = dataset.shape[0]

    pertinencia = np.random.rand(P, C)
    pertinencia = pertinencia / pertinencia.sum(axis=1, keepdims=True)

    assert pertinencia.sum() == P

    prototipos = relevancia = custo_atual = None

    for t in range(1, T + 1):
        prototipos = calcular_prototipos(dataset, pertinencia)
        relevancia = calcular_relevancia(dataset, pertinencia, prototipos, Tv)
        nova_pertinencia = calcular_pertinencia(dataset, relevancia, prototipos, Tu)

        assert nova_pertinencia.min() > 0
        assert nova_pertinencia.sum().round() == P
        assert relevancia.sum().round() == C

        custo_atual = funcao_objetivo(
            dataset, nova_pertinencia, relevancia, prototipos, Tu, Tv
        )
        print(f">> Ciclo {t}: {custo_atual}")

        if np.max(np.abs(nova_pertinencia - pertinencia)) < eps:
            pertinencia = nova_pertinencia
            break

        pertinencia = nova_pertinencia

        del nova_pertinencia

    return {
        "prototipos": prototipos,
        "pertinencia": pertinencia,
        "relevancia": relevancia,
        "custo": custo_atual,
    }


def executar_treinamentos(
    dataset, treinos=TOTAL_TREINAMENTOS, seed=SEED, T=T, Tu=TU, Tv=TV, eps=eps
):
    """

    :param dataset: Dataset desejado
    :param treinos: Total de treinos desejados
    :param seed: Semente para garantir reprodutibilidade
    :param T: Quantidade máxima de iterações
    :param Tu: Grau de fuzzificação
    :param Tv: Relevância das variáveis
    :param eps: Limitar de tolerência

    :return: dict (Melhor resultado obtido)

    """
    args = dict(T=T, Tu=Tu, Tv=Tv, eps=eps, dataset=dataset)
    melhor_resultado = {"custo": sys.maxsize}
    comp = itemgetter("custo")

    for t in range(treinos):
        print("> Treino", t + 1)
        resultado = executar_clusterizacao(seed=seed + t, **args)
        melhor_resultado = min(melhor_resultado, resultado, key=comp)

    # melhor_resultado["crisp"] = melhor_resultado["pertinencia"].argmax(axis=1)
    # melhor_resultado["particao"] = utils.get_hard_patitions_clustering(
    #     melhor_resultado["pertinencia"]
    # )
    # melhor_resultado["priori"] = (
    #     utils.calcular_probabilidade_priori(melhor_resultado["particao"]),
    # )
    return melhor_resultado


def exportar_resultado(dados, arquivo):
    with open(arquivo, "wb") as f:
        pickle.dump(dados, f)


def questa1(seed=SEED):
    import dataset

    PATH = f"{Path(__file__).parent.parent}/data/"

    for i, (name, dt) in enumerate(dataset.ALL_TEST.items()):
        # dt = minmax_scale(dt)
        # dt = scale(dt)
        dt = robust_scale(dt)
        resultado = executar_treinamentos(dt, seed=seed * i)
        print("Custo", name, ":", resultado["custo"])
        exportar_resultado(resultado, f"{PATH}/{name}.pickle")


if __name__ == "__main__":
    questa1()
