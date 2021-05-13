from dlc_practical_prologue import generate_pair_sets
import torch


def main():
    train_input, train_target, train_classes, \
        test_input, test_target, test_classes = generate_pair_sets(1000)

    # print(train_target.size())
    print(train_target)


if __name__ == '__main__':
    main()
