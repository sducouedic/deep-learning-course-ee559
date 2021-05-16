from dlc_practical_prologue import generate_pair_sets

from models import *
from plot import *


def main():
    number_rounds = 10

    model1 = Baseline(generate_pair_sets)
    mRes1 = model1.train_and_test_rounds(number_rounds)

    plot_model_result(mRes1, number_rounds)
    plot_models_results_comparison([mRes1], number_rounds, True)


if __name__ == '__main__':
    main()
