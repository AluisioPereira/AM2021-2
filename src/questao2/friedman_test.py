import pandas as pd
from scipy.stats import friedmanchisquare
import numpy as np


def get_scores():
    knn_scores = pd.read_csv("../../reports/knn_scores_for_friedman_test.csv")
    gaussian_scores = pd.read_csv("../../reports/gaussian_scores_for_friedman_test.csv")
    parzen_scores = pd.read_csv("../../reports/parzen_scores_for_friedman_test.csv")

    logistic_scores = pd.read_csv(
        "../../reports/logistic_regression_scores_for_friedman_test.csv"
    )

    knn_f1 = knn_scores["f1_weighted"].values
    gaussian_f1 = gaussian_scores["f1_weighted"].values
    parzen_f1 = parzen_scores["f1_weighted"].values
    logistic_f1 = logistic_scores["f1_weighted"].values

    return knn_f1, gaussian_f1, parzen_f1, logistic_f1


def friedman_test():
    knn_f1, gaussian_f1, parzen_f1, logistic_f1 = get_scores()

    result = friedmanchisquare(knn_f1, gaussian_f1, parzen_f1, logistic_f1)

    with open("../../reports/friendman_test.txt", "w") as f:
        f.write(str(result) + "\n")


def gerar_script_pos_teste():
    knn_f1, gaussian_f1, parzen_f1, logistic_f1 = get_scores()

    knn_f1 = knn_f1.round(3)
    gaussian_f1 = gaussian_f1.round(3)
    parzen_f1 = parzen_f1.round(3)
    logistic_f1 = logistic_f1.round(3)

    template = """#install.packages("coin")
library(coin)
#install.packages("multcomp")
library(multcomp)

#install.packages("PMCMR")
library(PMCMR)

x <- rep(c("gaussian", "knn", "parzen", "logistic"), 300)

x1 <- rep(c(1:300), each=4)
resp <- c({})

friedman.test(resp, groups=x, blocks=x1)
posthoc.friedman.nemenyi.test(resp, groups=x, blocks=x1)"""
    array = []
    for i, values in enumerate(zip(knn_f1, gaussian_f1, parzen_f1, logistic_f1)):
        if i < len(knn_f1) - 1:
            array.append(",".join(map(str, values)) + ",")
        else:
            array.append(",".join(map(str, values)))

    script = template.format("\n".join(array))

    with open("../../reports/friendman_test_pos_test.r", "w") as f:
        f.write(script)


def gerar_script_pos_teste2():
    import scikit_posthocs as sp

    knn_f1, gaussian_f1, parzen_f1, logistic_f1 = get_scores()
    data = np.array([knn_f1, gaussian_f1, parzen_f1, logistic_f1])

    print(sp.posthoc_nemenyi_friedman(data.T))


if __name__ == "__main__":
    friedman_test()
    gerar_script_pos_teste()
