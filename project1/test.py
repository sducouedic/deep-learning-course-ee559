from dlc_practical_prologue import generate_pair_sets

from models import Baseline


def main():
    # train_input, train_target, train_classes, \
    # test_input, test_target, test_classes = generate_pair_sets(1000)
    # print(train_input.size())

    model = Baseline(generate_pair_sets)
    model.train_and_test_round()


if __name__ == '__main__':
    main()
