from dlc_practical_prologue import generate_pair_sets

from models import *
from plot import *


def train_and_test_rounds(model, nb_rounds, test_id):
    """ Complete nb_rounds iterations of train_and_test and returns the train and test error
        rates and other performance statistics in an ModelResult object

        Parameter
        ---------
        nb_rounds : int
            the number of round

        Returns
        -------
        a ModelResult object containing the overall rounds performance statistics
    """

    trains_err_rates = []
    tests_err_rates = []
    losses = []

    for i in range(nb_rounds):
        model.reset()
        trains_err_rate, tests_err_rate, losses_ = model.train_and_test()
        trains_err_rates.append(trains_err_rate)
        tests_err_rates.append(tests_err_rate)
        if i == 0:
            losses += losses_

    return ModelResult(test_id, trains_err_rates, tests_err_rates, losses)


def main():
    number_rounds = 15

    # model1 = Baseline(generate_pair_sets)
    # mRes1 = train_and_test_rounds(model1, number_rounds, "Baseline")
    #
    # model2 = Auxiliary(generate_pair_sets)
    # mRes2 = train_and_test_rounds(model2, number_rounds, "Auxiliary")

    # model1 = CNN(generate_pair_sets)
    # model2 = CNN(generate_pair_sets)
    model3 = CNN(generate_pair_sets)
    mRes3 = train_and_test_rounds(model3, number_rounds, "CNN")

    # model4 = CNN_Auxiliary(generate_pair_sets)
    # mRes4 = train_and_test_rounds(model4, number_rounds, "CNN_Aux")

    plot_model_result(mRes3, number_rounds)
    # plot_models_results_comparison([mRes1, mRes2, mRes3, mRes4], number_rounds, True)

    # plot_model_result(mRes2, number_rounds)
    # plot_models_results_comparison([mRes1, mRes2], number_rounds, True)


if __name__ == '__main__':
    main()
