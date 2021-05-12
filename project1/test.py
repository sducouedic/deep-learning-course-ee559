from dlc_practical_prologue import generate_pair_sets
import torch


def main():
    train_set, train_input, train_target, test_set, test_input, test_target = generate_pair_sets(1000)


if __name__ == '__main__':
    main()
