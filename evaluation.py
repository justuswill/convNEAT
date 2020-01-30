import logging
import torch

from net import Net


def evaluate_genome_on_data(genome, torch_device, data_loader_train, data_loader_test, input_size,
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
        monitor.send([1, [(genome.__class__, genome.save())],
                      {'kind': 'net-plot', 'n': n, 'i': i, 'title': 'train', 'input_size': input_size}])

    logging.debug('Building Net')
    net = Net(genome, input_size=input_size)
    net = net.to(torch_device)
    criterion = torch.nn.CrossEntropyLoss()
    try:
        optimizer = torch.optim.SGD(net.parameters(), lr=2**genome.log_learning_rate, momentum=.9)
    except ValueError:
        import pdb;pdb.set_trace()

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

    class_total = list(0 for i in range(10))
    class_correct = list(0 for i in range(10))
    with torch.no_grad():
        for inputs, labels in data_loader_test:
            outputs = net(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions = predictions == labels
            for label, correct in zip(labels, correct_predictions):
                class_total[label] += 1
                class_correct[label] += correct.item()
    for i in range(10):
        print('Accuracy of {}: {:5.2f} %  ({} / {})'.format(
                i, 100 * class_correct[i] / class_total[i], class_correct[i],
                class_total[i]))

    total = sum(class_total)
    correct = sum(class_correct)
    print('Accuracy of the network on the {} test images: {:5.2f} %  '
          '({} / {})'.format(total, 100 * correct / total, correct, total))
    print()

    return correct / total