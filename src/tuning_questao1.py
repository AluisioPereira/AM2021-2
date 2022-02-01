import optuna
from optuna.multi_objective.samplers import NSGAIIMultiObjectiveSampler
from optuna.samplers import TPESampler
import sys
from questao1 import executar_clusterizacao
import dataset
import utils


def get_objetctive(dataset):
    def objective(trial):
        Tu = trial.suggest_float("Tu", 0.01, 100)
        Tv = trial.suggest_float("Tv", 10, 1e8)

        try:
            res = executar_clusterizacao(dataset, seed=1, Tu=Tu, Tv=Tv)
            custo = res["custo"]
            particao, _ = utils.get_hard_patitions_clustering(res["pertinencia"])
            clusters_nao_vazios = sum([1 for part in particao if part])
            return custo, clusters_nao_vazios
        except:
            return sys.maxsize, 0

    return objective


def run_study(dataset_name, dataset, n_trials=1000):
    study = optuna.multi_objective.create_study(
        ["minimize", "maximize"],
        study_name="dataset_name",
        storage=f"sqlite:///{dataset_name}_study.db",
        load_if_exists=True,
        sampler=NSGAIIMultiObjectiveSampler(seed=1),
    )

    # study = optuna.create_study(
    #     study_name="dataset_name",
    #     storage=f"sqlite:///{dataset_name}_study.db",
    #     load_if_exists=True,
    #     sampler=TPESampler(),
    # )

    study.optimize(get_objetctive(dataset), n_trials=n_trials, n_jobs=2)
    df = study.trials_dataframe()
    df.to_csv(f"{dataset_name}_study.csv")


if __name__ == "__main__":
    run_study("rgb_test", dataset.RGB_TEST)
