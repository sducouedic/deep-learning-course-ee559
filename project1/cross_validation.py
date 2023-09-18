# -*- coding: utf-8 -*-
"""
Created on Wed May 26 12:02:35 2021

@author: mauro
"""

# ======================================================================================
# ------------------------------------ Disclaimer --------------------------------------
# This is the 2-fold cross-validation script we used to get the best possible parameters.
# However the architectures were adapted especially for this script to be able to run
# which is no more the case. Do not run this file as it won't produce the expected results
# we added it only for you to see our work.
# --------------------------------------------------------------------------------------
# ======================================================================================


from models import *


def two_param_twofold_crossvalidate(modelName, min_lr, max_lr, min_bs, max_bs, train_data,
                                    train_target, train_classes):
    lenght = 1000
    folds = 2
    final_lr = 0
    final_bs = 0
    minloss = 0
    tot_loss = 0
    number_of_possibilities = 10
    learning_rates = []
    for i in range(number_of_possibilities):
        learning_rates.append(min_lr + (max_lr - min_lr) * i / (number_of_possibilities - 1))
    number_of_possibilities = 10
    batch_sizes = []
    for i in range(number_of_possibilities):
        batch_sizes.append(min_bs + (max_bs - min_bs) * i / (number_of_possibilities - 1))

    train1 = train_data[:int(lenght / 2)]
    train2 = train_data[int(lenght / 2):]
    train1tgt = train_target[:int(lenght / 2)]
    train2tgt = train_target[int(lenght / 2):]
    train1class = train_classes[int(lenght / 2):]
    train2class = train_classes[:int(lenght / 2)]
    trclasses = [train1class, train2class]
    tstclasses = [train2class, train1class]
    trains = [train1, train2]
    tests = [train2, train1]
    trainstgt = [train1tgt, train2tgt]
    teststgt = [train2tgt, train1tgt]

    minloss = float('inf')
    tot_loss = 0
    final_lr = 0
    final_bs = 0
    loss = 0
    for bs in batch_sizes:
        for lr in learning_rates:
            print('testing for (bs,lr): ', bs, lr)
            for i in range(folds):
                if modelName == "Baseline":
                    model = Baseline(None, 25, 100, lr, bs)
                elif modelName == "Auxiliary":
                    model = Base_Aux(None, 25, 100, lr, bs)
                elif modelName == "CNN":
                    model = CNN(None, 25, 100, lr, bs)
                elif modelName == "CNN_Auxiliary":
                    model = CNN_Aux(None, 25, 100, lr, bs)
                model._train(trains[i], trainstgt[i], )
                loss = model._compute_errors(tests[i], teststgt[i], trclasses)
                tot_loss += loss
            if tot_loss < minloss:
                minloss = tot_loss
                final_lr = lr
                final_bs = bs
            tot_loss = 0
    print('final_lr: ', final_lr, 'final_bs: ', final_bs, 'minloss', minloss)
    return


def one_param_twofold_crossvalidate(modelName, min_lr, max_lr, train_data, train_target,
                                    train_classes):
    lenght = 1000
    folds = 2
    final_lr = 0
    final_bs = 0
    minloss = 0
    tot_loss = 0
    number_of_possibilities = 50
    learning_rates = []
    for i in range(number_of_possibilities):
        learning_rates.append(min_lr + (max_lr - min_lr) * i / (number_of_possibilities - 1))
    number_of_possibilities = 10

    train1 = train_data[:int(lenght / 2)]
    train2 = train_data[int(lenght / 2):]
    train1tgt = train_target[:int(lenght / 2)]
    train2tgt = train_target[int(lenght / 2):]
    train1class = train_classes[int(lenght / 2):]
    train2class = train_classes[:int(lenght / 2)]
    trclasses = [train1class, train2class]
    tstclasses = [train2class, train1class]
    trains = [train1, train2]
    tests = [train2, train1]
    trainstgt = [train1tgt, train2tgt]
    teststgt = [train2tgt, train1tgt]

    minloss = float('inf')
    tot_loss = 0
    final_lr = 0
    final_bs = 0
    loss = 0

    for lr in learning_rates:
        print('testing for (lr): ', lr)
        for i in range(folds):
            if modelName == "Baseline":
                model = Baseline(None, 25, 100, lr)
            elif modelName == "Auxiliary":
                model = Auxiliary(None, 25, 100, lr)
            elif modelName == "CNN":
                model = CNN(None, 25, 100, lr)
            elif modelName == "CNN_Auxiliary":
                model = CNN_Auxiliary(None, 25, 100, lr)

            model._train(trains[i], trainstgt[i], trclasses[i])
            loss = model._compute_errors(tests[i], teststgt[i])
            tot_loss += loss
        print('total loss: ', tot_loss)
        if tot_loss < minloss:
            minloss = tot_loss
            final_lr = lr
        tot_loss = 0
    print('final_lr: ', final_lr, 'final_bs: ', final_bs, 'minloss', minloss)
    return