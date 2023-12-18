def train_cnn(
        model,
        input_tensor,
        output_tensor,
        epochs,
        batch_size,
        lossFunction,
        optimizer,
        print_loss=False
):
    losses_train = []
    for i in range(epochs):
        losses_batch = []
        for b in range(batch_size):
            y_pred = model.forward(input_tensor[b])

            loss_train = lossFunction(y_pred, output_tensor[b])
            losses_batch.append(loss_train.detach().numpy())

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        losses_train.append(losses_batch)

        if print_loss:
            print(f'Epoch: {i} / loss: {loss_train}')

    return model, losses_train