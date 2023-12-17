def train_cnn(
        model,
        input_tensor,
        output_tensor,
        epochs,
        lossFunction,
        optimizer,
        print_loss=False
):
    losses_train = []
    for i in range(epochs):
        y_pred = model.forward(input_tensor)

        loss_train = lossFunction(y_pred, output_tensor)
        losses_train.append(loss_train.detach().numpy())

        if(print_loss):
            if i%10 == 0:
                print(f'Epoch: {i} / Loss: {loss_train}')

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()