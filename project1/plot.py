import cmath

import matplotlib.pyplot as plt


class ModelResult:
    def __init__(self, name, train_err_rates, test_err_rates, losses):
        self.name = name
        self.train_err_rates = train_err_rates
        self.test_err_rates = test_err_rates
        self.losses = losses


# --------------------------------------------------------------------------------#

colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple',
          'tab:yellow', 'tab:black', 'tab:skyblue', 'tab:chocolate', 'tab:lawngreen']


def plot_models_results_comparison(model_results, nb_rounds, means_only=False):
    """ Plot all the models performances statistics (images are saved in "results" folder """
    if not means_only:
        plot_train_err_rates(model_results, nb_rounds)
        plot_test_err_rates(model_results, nb_rounds)

    plot_train_err_rates_means(model_results)
    plot_test_err_rates_means(model_results)
    plot_losses(model_results)


def plot_model_result(model_result, nb_rounds):
    """ Plot the statistics for a single model (images are saved in "results" folder) """

    plot_single_model_train_test_err_rates_comparison(model_result, nb_rounds)
    plot_losses([model_result])


# --------------------------------------- Local Use ---------------------------------------#

def plot_train_err_rates(model_results, nb_rounds):
    """ Plot the train error rates of the models in a single chart """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Round', labelpad=nb_rounds)
    ax.set_ylabel('Train error rate')

    max_err_rate = max([max(mod_res.train_err_rates) for mod_res in model_results])
    ax.set_ylim([0, max_err_rate + 0.05 * max_err_rate])

    for i, mod_res in enumerate(model_results):
        ax.plot(mod_res.train_err_rates, color=colors[i % len(colors)], label=mod_res.name)

    plt.savefig("results/train_err" + concat_models_names(model_results) + ".png")
    plt.show()


def plot_train_err_rates_means(model_results):
    """ Plot the train error rates of the models in a single chart """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_ylabel('Train error rate')

    names = [mod_res.name for mod_res in model_results]
    means = [mean(mod_res.train_err_rates) for mod_res in model_results]
    stds = [std(mod_res.train_err_rates) for mod_res in model_results]

    ax.bar(names, means, color=colors[0:len(model_results)], yerr=stds, capsize=4)

    plt.savefig("results/train_err_avg" + concat_models_names(model_results) + ".png")
    plt.show()


def plot_test_err_rates(model_results, nb_rounds):
    """ Plot the test error rates of the models in a single chart """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Round', labelpad=nb_rounds)
    ax.set_ylabel('Test error rate')

    max_err_rate = max([max(mod_res.test_err_rates) for mod_res in model_results])
    ax.set_ylim([0, max_err_rate + 0.05 * max_err_rate])

    for i, mod_res in enumerate(model_results):
        ax.plot(mod_res.test_err_rates, color=colors[i % len(colors)], label=mod_res.name)

    plt.savefig("results/test_err" + concat_models_names(model_results))


def plot_test_err_rates_means(model_results):
    """ Plot the train error rates of the models in a single chart """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_ylabel('Test error rate')

    names = [mod_res.name for mod_res in model_results]
    means = [mean(mod_res.test_err_rates) for mod_res in model_results]
    stds = [std(mod_res.test_err_rates) for mod_res in model_results]

    ax.bar(names, means, color=colors[0:len(model_results)], yerr=stds, capsize=4)

    plt.savefig("results/test_err_avg" + concat_models_names(model_results) + ".png")
    plt.show()


def plot_single_model_train_test_err_rates_comparison(model_result, nb_rounds):
    """ Plot the test and train error rates for a single model """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Round', labelpad=nb_rounds)
    ax.set_ylabel('Error rate')

    max_err_rate = max(max(model_result.train_err_rates), max(model_result.test_err_rates))
    ax.set_ylim([0, max_err_rate + 0.05 * max_err_rate])

    ax.plot(model_result.train_err_rates, color=colors[0], label="train")
    ax.plot(model_result.test_err_rates, color=colors[1], label="test")

    plt.savefig("results/train_test_err" + model_result.name)
    plt.show()


def plot_losses(model_results):
    """ Plot the loss evolution """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('Iteration', labelpad=len(model_results[0].losses))
    ax.set_ylabel('Loss')

    max_err_rate = max([max(mod_res.losses) for mod_res in model_results])
    ax.set_ylim([0, max_err_rate + 0.05 * max_err_rate])

    for i, mod_res in enumerate(model_results):
        ax.plot(mod_res.losses, color=colors[i % len(colors)], label=mod_res.name)

    plt.savefig("results/losses_evolution" + concat_models_names(model_results))
    plt.show()


def concat_models_names(model_results):
    name = ""
    for mod_res in model_results:
        name += "_" + mod_res.name
    return name


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) == 0:
        return 0

    mean_ = mean(values)
    variance = sum([(v - mean_) ** 2 for v in values]) / (len(values) - 1)
    return cmath.sqrt(variance).real
