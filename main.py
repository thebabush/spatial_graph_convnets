#!/usr/bin/env python3

import time

import numpy as np
import sklearn.metrics
import torch

import config
import sgraphnn.model
import sgraphnn.util.graph_generator


if torch.cuda.is_available():
     print('cuda available')
     dtypeLong = torch.cuda.LongTensor
     if config.CONSTANT_SEED:
         torch.cuda.manual_seed(1)
else:
     print('cuda not available')
     dtypeLong = torch.LongTensor
     if config.CONSTANT_SEED:
         torch.manual_seed(1)


#################
# network and optimization parameters
#################

# network parameters
net_parameters = {}
net_parameters['Voc'] = config.task_parameters['Voc']
net_parameters['D'] = 50
net_parameters['nb_clusters_target'] = config.task_parameters['nb_clusters_target']
net_parameters['H'] = 50
net_parameters['L'] = 10

# optimization parameters
opt_parameters = {}
opt_parameters['learning_rate'] = 0.00075   # ADAM
opt_parameters['max_iters'] = 500
opt_parameters['batch_iters'] = 100
opt_parameters['decay_rate'] = 1.25


#########################
# Graph convnet function
#########################
def our_graph_convnets(task_parameters, net_parameters, opt_parameters):
     # Delete existing network if exists
    try:
        del net
        print('Delete existing network\n')
    except NameError:
        print('No existing network to delete\n')

    # instantiate
    net = sgraphnn.model.Graph_OurConvNet(
            net_parameters['Voc'],
            net_parameters['D'],
            net_parameters['nb_clusters_target'],
            net_parameters['H'],
            net_parameters['L'],
    )
    if torch.cuda.is_available():
        net.cuda()
    print(net)

    # number of network parameters
    nb_param = 0
    for param in net.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('nb_param=',nb_param,' L=',net_parameters['L'])

    # network parameters
    Voc = net_parameters['Voc']
    D = net_parameters['D']
    nb_clusters_target = net_parameters['nb_clusters_target']
    H = net_parameters['H']
    L = net_parameters['L']
    # optimization parameters
    learning_rate = opt_parameters['learning_rate']
    max_iters = opt_parameters['max_iters']
    batch_iters = opt_parameters['batch_iters']
    decay_rate = opt_parameters['decay_rate']

    # Optimizer
    global_lr = learning_rate
    global_step = 0
    lr = learning_rate
    optimizer = net.update(lr)


    #############
    # loop over epochs
    #############
    t_start = time.time()
    t_start_total = time.time()
    average_loss_old = 1e10
    running_loss = 0.0
    running_total = 0
    running_conf_mat = 0
    running_accuracy = 0
    tab_results = []
    for iteration in range(1*max_iters):  # loop over the dataset multiple times
        # generate one train graph
        train_x = sgraphnn.util.graph_generator.variable_size_graph(task_parameters)

        train_y = train_x.target
        train_y = torch.autograd.Variable( torch.LongTensor(train_y).type(dtypeLong) , requires_grad=False)

        # forward, loss
        y = net.forward(train_x)
        # compute loss weigth
        labels = train_y.data.cpu().numpy()
        V = labels.shape[0]
        nb_classes = len(np.unique(labels))
        cluster_sizes = np.zeros(nb_classes)
        for r in range(nb_classes):
            cluster = np.where(labels==r)[0]
            cluster_sizes[r] = len(cluster)
        weight = torch.zeros(nb_classes)
        for r in range(nb_classes):
            sumj = 0
            for j in range(nb_classes):
                if j!=r:
                    sumj += cluster_sizes[j]
            weight[r] = sumj/ V
        loss = net.loss(y,train_y,weight)
        loss_train = loss.data.item()
        running_loss += loss_train
        running_total += 1

        # confusion matrix
        S = train_y.data.cpu().numpy()
        C = np.argmax( torch.nn.Softmax(dim=0)(y).data.cpu().numpy() , axis=1)
        CM = sklearn.metrics.confusion_matrix(S,C).astype(np.float32)
        nb_classes = CM.shape[0]
        train_y = train_y.data.cpu().numpy()
        for r in range(nb_classes):
            cluster = np.where(train_y==r)[0]
            CM[r,:] /= cluster.shape[0]
        running_conf_mat += CM
        running_accuracy += np.sum(np.diag(CM))/ nb_classes

        # backward, update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # learning rate, print results
        if iteration % batch_iters == 0:

            # time
            t_stop = time.time() - t_start
            t_start = time.time()

            # confusion matrix
            average_conf_mat = running_conf_mat/ running_total
            running_conf_mat = 0

            # accuracy
            average_accuracy = running_accuracy/ running_total
            running_accuracy = 0

            # update learning rate
            average_loss = running_loss/ running_total
            if average_loss > 0.99* average_loss_old:
                lr /= decay_rate
            average_loss_old = average_loss
            optimizer = net.update_learning_rate(optimizer, lr)
            running_loss = 0.0
            running_total = 0

            # save intermediate results
            tab_results.append([iteration,average_loss,100* average_accuracy,time.time()-t_start_total])

            # print results
            print('iteration={:3d}, loss={:.3f}, lr={:.8f}, time={:.3}'.format(
                iteration,
                average_loss,
                lr,
                t_stop,
                ))
            #print('Confusion matrix= \n', 100* average_conf_mat)
            print('accuracy= %.3f' % (100* average_accuracy))

    ############
    # Evaluation on 100 pre-saved data
    ############
    running_loss = 0.0
    running_total = 0
    running_conf_mat = 0
    running_accuracy = 0
    for iteration in range(100):

        # generate one data
        train_x = sgraphnn.util.graph_generator.variable_size_graph(task_parameters)
        train_y = train_x.target
        train_y = torch.autograd.Variable( torch.LongTensor(train_y).type(dtypeLong) , requires_grad=False)

        # forward, loss
        y = net.forward(train_x)
        # compute loss weigth
        labels = train_y.data.cpu().numpy()
        V = labels.shape[0]
        nb_classes = len(np.unique(labels))
        cluster_sizes = np.zeros(nb_classes)
        for r in range(nb_classes):
            cluster = np.where(labels==r)[0]
            cluster_sizes[r] = len(cluster)
        weight = torch.zeros(nb_classes)
        for r in range(nb_classes):
            sumj = 0
            for j in range(nb_classes):
                if j!=r:
                    sumj += cluster_sizes[j]
            weight[r] = sumj/ V
        loss = net.loss(y,train_y,weight)
        loss_train = loss.data.item()
        running_loss += loss_train
        running_total += 1

        # confusion matrix
        S = train_y.data.cpu().numpy()
        C = np.argmax( torch.nn.Softmax(dim=0)(y).data.cpu().numpy() , axis=1)
        CM = sklearn.metrics.confusion_matrix(S,C).astype(np.float32)
        nb_classes = CM.shape[0]
        train_y = train_y.data.cpu().numpy()
        for r in range(nb_classes):
            cluster = np.where(train_y==r)[0]
            CM[r,:] /= cluster.shape[0]
        running_conf_mat += CM
        running_accuracy += np.sum(np.diag(CM))/ nb_classes

        # confusion matrix
        average_conf_mat = running_conf_mat/ running_total
        average_accuracy = running_accuracy/ running_total
        average_loss = running_loss/ running_total

    # print results
    print('\nloss(100 pre-saved data)= %.3f, accuracy(100 pre-saved data)= %.3f' % (average_loss,100* average_accuracy))


    #############
    # output
    #############
    result = {}
    result['final_loss'] = average_loss
    result['final_acc'] = 100* average_accuracy
    result['final_CM'] = 100* average_conf_mat
    result['final_batch_time'] = t_stop
    result['nb_param_nn'] = nb_param
    result['plot_all_epochs'] = tab_results
    return result


if __name__ == '__main__':
    results = our_graph_convnets(config.task_parameters, net_parameters, opt_parameters)
    print('=' * 80)
    print('RESULTS:')
    for k, v in results.items():
        print('\t{}: {}'.format(k, v))

