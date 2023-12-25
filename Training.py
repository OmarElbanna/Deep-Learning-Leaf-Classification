from Model_Evaluation import evaluate_model_with_outputs

def train_cnn(
        model,
        images_train,
        labels_train,
        images_test,
        labels_test,
        epochs,
        batch_size,
        lossFunction,
        optimizer,
        print_loss=False,
        calc_accuracy=False
):
    losses_train = []
    accuracies_train = []
    accuracies_test = []

    for i in range(epochs):
        losses_batch = []
        for b in range(len(images_train)):
            y_pred = model.forward(images_train[b])

            loss_train = lossFunction(y_pred, labels_train[b])
            losses_batch.append(loss_train.detach().numpy())

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        losses_train.append(losses_batch)

        if print_loss:
            print(f'Epoch: {i} / loss: {loss_train}')

        if calc_accuracy:
            accuracies_train.append(
                evaluate_model_with_outputs(model, images_train, labels_train, batch_size, 0) * 100
            )
            accuracies_test.append(
                evaluate_model_with_outputs(model, images_test, labels_test, batch_size, 1) * 100
            )

    return model, losses_train, accuracies_train, accuracies_test