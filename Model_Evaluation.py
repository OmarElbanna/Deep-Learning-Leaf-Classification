import torch
import matplotlib.pyplot as plt
def evaluate_model_with_outputs(model, inputs, outputs, batch_size, evaluation_type:bool):
    """
    Inputs:
      model: The PyTorch model to be evaluated.
      inputs: List of input tensors.
      outputs: List of model output tensors.
      evaluation_type: 0 in case of training, 1 in testing

    """
    model.eval()  # Set the model to evaluation mode

    correct_predictions = 0
    samples_in_epoch = 0
    correct_predictions_in_epochs = 0
    accuracy_in_epoch = 0
    accuracy_list =[]
    num_epochs = 0

    if (evaluation_type == 0):
      total_samples = (len(inputs)-1)*batch_size+len(inputs[len(inputs)-1])

      with torch.no_grad():  # Disable gradient calculation during evaluation
        for input_tensor, output_tensor in zip(inputs, outputs):

          num_epochs+=1
          samples_in_epoch = len(input_tensor)

          predictions = model(input_tensor)
          _, predicted_classes = torch.max(predictions, 1)

          correct_predictions_in_epochs = (predicted_classes == output_tensor).sum().item()
          accuracy_in_epoch = correct_predictions_in_epochs/samples_in_epoch
          accuracy_list.append(accuracy_in_epoch)

          correct_predictions += correct_predictions_in_epochs

      # Plot the accuracy over epochs
      plt.plot(range(1, num_epochs+1), accuracy_list, marker='o')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.title('Accuracy vs Epoch')
      plt.grid(True)
      plt.show()


    else:
      total_samples=len(inputs)

      with torch.no_grad():  # Disable gradient calculation during evaluation
          # Forward pass
          predictions = model(inputs)

          # Assuming outputs contain ground truth labels
          _, predicted_classes = torch.max(predictions, 1)
          correct_predictions = (predicted_classes == outputs).sum().item()

    accuracy = correct_predictions / total_samples
    print(f"{evaluation_type} Model Accuracy: {accuracy * 100:.2f}%")