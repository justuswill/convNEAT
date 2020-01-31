import logging
import torch
import numpy as np

from net import Net
from optimizer import SGDGene, ADAMGene


def evaluate_genome_on_data(genome, torch_device, data_loader_train, data_loader_test, input_size, output_size,
                            epochs=2, monitor=None):
    """
    Instantiate the neural network from the genome and train it for a set amount of epochs
    Evaluate the accuracy on the test data and return this as the score.
    If a monitor is set, visualize the net that is currently training.
    """

    n = genome.population.n
    i = next(genome.population.i) + 1

    print('Instantiating neural network from the following genome(%d/%d):' % (i, n))
    print(genome)
    # Visualize current net
    if monitor is not None:
        monitor.plot(1, (genome.__class__, genome.save()), kind='net-plot', title='train',
                     n=n, i=i, input_size=input_size)

    logging.debug('Building Net')
    net = Net(genome, input_size=input_size, output_size=output_size)

    # Load saved parameters
    net_dict = net.state_dict()
    all_net_parameters = {k: v for gene in genome.genes for k, v in gene.net_parameters.items()
                          if k in net_dict}
    net_dict.update(all_net_parameters)
    net.load_state_dict(net_dict)

    net = net.to(torch_device)
    criterion = torch.nn.CrossEntropyLoss()

    opt = genome.optimizer
    if type(opt) == SGDGene:
        optimizer = torch.optim.SGD(net.parameters(), lr=2**opt.log_learning_rate, momentum=opt.momentum,
                                    weight_decay=2**opt.log_weight_decay, nesterov=True)
    elif type(opt) == ADAMGene:
        optimizer = torch.optim.Adam(net.parameters(), lr=2**opt.log_learning_rate,
                                     weight_decay=2**opt.log_weight_decay)
    else:
        raise ValueError('Optimizer %s not supported' % type(genome.optimizer))

    print('Beginning training')
    for epoch in range(epochs):
        epoch_loss = 0.
        batch_loss = 0.
        n = len(data_loader_train) // 10
        for i, (inputs, labels) in enumerate(data_loader_train):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_loss += loss.item()
            if (i+1) % n == 0:
                print('[{}, {:3}] loss: {:.3f}'.format(
                        epoch, i+1, batch_loss / n))
                batch_loss = 0.
        print('[{}] loss: {:.3f}'.format(
                epoch, epoch_loss / len(data_loader_train)))
    print('Finished training')

    confusion = np.zeros((output_size, output_size))
    with torch.no_grad():
        for inputs, labels in data_loader_test:
            outputs = net(inputs)
            predictions = torch.argmax(outputs, dim=1)
            for pre, lab in zip(predictions, labels):
                confusion[lab, pre] += 1

    print(confusion)
    class_total = np.sum(confusion, axis=1)
    labeled_total = np.sum(confusion, axis=0)
    class_correct = confusion.diagonal()
    correct = int(np.sum(class_correct))
    total = int(np.sum(class_total))
    acc = correct / total

    # Ignore Warning if /0
    np.seterr(divide='ignore', invalid='ignore')
    for i in range(output_size):
        print('%d: Recall: %5.2f %%  (%3d / %3d) - Precision %5.2f %% (%3d / %3d)' %
              (i, 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i],
               100 * class_correct[i] / labeled_total[i], class_correct[i], labeled_total[i]))
    print('Accuracy of the network on the %d validation images: %5.2f %% (%d / %d)' % (total, 100 * acc, correct, total))
    print()
    np.seterr(divide='warn', invalid='warn')

    # Save weights and bias
    for name, parameter in net.state_dict().items():
        if name.startswith('conv') or name.startswith('pool_'):
            _id = int(name.split('.')[0].split('_')[-1])
            genome.genes_by_id[_id].net_parameters[name] = parameter.to('cpu')

    return acc
