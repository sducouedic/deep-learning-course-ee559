from dlc_practical_prologue import generate_pair_sets

from models import Baseline
from plot import *


def main():
    number_rounds = 10

    model = Baseline(generate_pair_sets)
    mRes = model.train_and_test_rounds(number_rounds)

    plot_model_result(mRes, number_rounds)
    plot_models_results_comparison([mRes], number_rounds, True)


if __name__ == '__main__':
    main()
