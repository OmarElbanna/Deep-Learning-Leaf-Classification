import torch

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
    if (evaluation_type == 0):
      total_samples = (len(inputs)-1)*batch_size+len(inputs[len(inputs)-1])
      with torch.no_grad():  # Disable gradient calculation during evaluation
        for input_tensor, output_tensor in zip(inputs, outputs):

            predictions = model(input_tensor)
            # torch.max returns a tuple of 2 elements
            # The first tuple is the value of the max between the output & 1
            # Which is intentionally ignored
            # The second tuple is the index of the max value
            _, predicted_classes = torch.max(predictions, 1)
            correct_predictions += (predicted_classes == output_tensor).sum().item()


    else:
      total_samples=len(inputs)

      with torch.no_grad():  # Disable gradient calculation during evaluation
          # Forward pass
          predictions = model(inputs)

          # Assuming outputs contain ground truth labels
          _, predicted_classes = torch.max(predictions, 1)
          correct_predictions = (predicted_classes == outputs).sum().item()

    accuracy = correct_predictions / total_samples

    return accuracy