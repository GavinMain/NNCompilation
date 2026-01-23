import numpy as np
import time
import torch
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#Hyper Parameters
num_epochs = 1
lr = .001
batch_size = 64

#Other Parameters
log_file = "log.txt"

def relu(x):
    return np.maximum(0, x)

def relu_gradient(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    result = exp_x / sum_exp_x
    return result

def cross_entropy_loss(predictions, targets, eps=1e-12):
    predictions = np.clip(predictions, eps, 1.0)
    loss = -np.sum(targets * np.log(predictions), axis=1)
    return np.mean(loss)

def cross_entropy_loss_gradient(predictions, targets):
    gradient = predictions - targets
    return gradient

class LinearLayer:
  def __init__(self, input_size, output_size):
      self.weights =np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
      self.bias = np.zeros((1, output_size))

  def forward(self, input_data):
      self.input = input_data

      self.out = np.dot(self.input, self.weights) + self.bias
      return self.out

  def backward(self, gradient, learning_rate):
      weights_gradient = np.dot(self.input.T, gradient) / self.input.shape[0]
      bias_gradient = np.sum(gradient, axis=0, keepdims=True) / self.input.shape[0]

      self.weights -= learning_rate * weights_gradient
      self.bias -= learning_rate * bias_gradient

      return np.dot(gradient, self.weights.T)

def train_model(model, train_data, train_labels, learning_rate, batch_size):
  for i in range(0, len(train_data), batch_size):
    batch_data = train_data[i:i+batch_size]
    batch_labels = train_labels[i:i+batch_size]
    out = model.forward(batch_data)
    predictions = softmax(out)
    loss = cross_entropy_loss(predictions, batch_labels)
    gradient = cross_entropy_loss_gradient(predictions, batch_labels)
    model.backward(gradient, learning_rate)
    break
  return np.average(loss)

def evaluate_model(model, test_data, test_labels):
  return 0
  out = model.forward(test_data)
  out = softmax(out)
  predictions = np.argmax(out, axis=1)
  true_labels = np.argmax(test_labels, axis=1)
  correct_predictions = np.sum(predictions == true_labels)
  return correct_predictions / len(test_labels)

class LayerNorm:
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))  

    def forward(self, input_data):
        self.input = input_data
        self.mean = np.mean(input_data, axis=1, keepdims=True)
        self.var = np.var(input_data, axis=1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)

        self.norm = (input_data - self.mean) / self.std
        out = self.gamma * self.norm + self.beta
        return out

    def backward(self, grad_output, learning_rate=1e-3):
        _, num_features = grad_output.shape

        gamma_gradient = np.sum(grad_output * self.norm, axis=0, keepdims=True)
        beta_gradient = np.sum(grad_output, axis=0, keepdims=True)

        norm_gradient = grad_output * self.gamma
        var_gradient = np.sum(norm_gradient * (self.input - self.mean) * -0.5 * (self.var + self.eps)**-1.5, axis=1, keepdims=True)
        mean_gradient = np.sum(norm_gradient * -1 / self.std, axis=1, keepdims=True) + var_gradient * np.mean(-2 * (self.input - self.mean), axis=1, keepdims=True)

        return_gradient = norm_gradient / self.std + var_gradient * 2 * (self.input - self.mean) / num_features + mean_gradient / num_features

        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient

        return return_gradient

class ConvolutionLayer2D:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.weights = np.random.uniform(-1, 1, (out_channels, in_channels, kernel_size, kernel_size))
    self.bias = np.zeros(out_channels)

  def forward(self, input_data):
    self.input = input_data
    batch_size, _, h, w = input_data.shape

    out_height = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

    if self.padding > 0:
        input_padded = np.pad(
            input_data,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant'
        )
    else:
        input_padded = input_data

    output = np.zeros((batch_size, self.out_channels, out_height, out_width))

    for b in range(batch_size):
        for oc in range(self.out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size

                    region = input_padded[b, :, h_start:h_end, w_start:w_end]
                    output[b, oc, i, j] = np.sum(region * self.weights[oc]) + self.bias[oc]

    self.out = output
    return output

  def backward(self, gradient, learning_rate):
    batch_size, _, h, w = self.input.shape

    if self.padding > 0:
        input_padded = np.pad(
            self.input,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant'
        )
    else:
        input_padded = self.input

    input_gradient_padded = np.zeros_like(input_padded)
    weights_gradient = np.zeros_like(self.weights)
    bias_gradient = np.zeros_like(self.bias)

    out_height, out_width = gradient.shape[2], gradient.shape[3]

    for b in range(batch_size):
        for oc in range(self.out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size

                    region = input_padded[b, :, h_start:h_end, w_start:w_end]

                    weights_gradient[oc] += gradient[b, oc, i, j] * region
                    input_gradient_padded[b, :, h_start:h_end, w_start:w_end] += gradient[b, oc, i, j] * self.weights[oc]

            bias_gradient[oc] += np.sum(gradient[b, oc])
            
    if self.padding > 0:
        return_gradient = input_gradient_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    else:
        return_gradient = input_gradient_padded

    self.weights -= learning_rate * (weights_gradient / batch_size)
    self.bias -= learning_rate * (bias_gradient / batch_size)

    return return_gradient

class CNN:
  def __init__(self):
      self.conv1 = ConvolutionLayer2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
      self.conv2 = ConvolutionLayer2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
      self.linear = LinearLayer(32 * 28 * 28, 10)

  def forward(self, input_data):
      input_data = self.conv1.forward(input_data) 
      input_data = relu(input_data)
      input_data = self.conv2.forward(input_data) 
      input_data = relu(input_data)
      
      self.flatten_shape = input_data.shape  
      input_data = input_data.reshape(input_data.shape[0], -1)
      input_data = self.linear.forward(input_data)
      return input_data

  def backward(self, gradient, learning_rate):
      gradient = self.linear.backward(gradient, learning_rate)  
      gradient = gradient.reshape(self.flatten_shape)
      
      gradient = gradient * relu_gradient(self.conv2.out)
      gradient = self.conv2.backward(gradient, learning_rate)
      gradient = gradient * relu_gradient(self.conv1.out)
      gradient = self.conv1.backward(gradient, learning_rate)

      return gradient

if __name__ == "__main__":
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  print("Images Before Processing:")
  print("Train Shape:", train_images.shape)
  print("Test Shape:", test_images.shape)
  print("Type:", train_images.dtype)
  print("Min, Max:", np.min(train_images), np.max(train_images))

  #Converts type to float64 and Scale to max=1. No Flatten
  train_images = train_images.reshape((train_images.shape[0], 1, 28, 28)).astype('float64') / 255
  test_images = test_images.reshape((test_images.shape[0], 1, 28, 28)).astype('float64') / 255

  print("\nImages After Processing:")
  print("Train Shape:", train_images.shape)
  print("Test Shape:", test_images.shape)
  print("Type:", train_images.dtype)
  print("Min, Max:", np.min(train_images), np.max(train_images))

  print("\nLabels Before Processing:")
  print("Train Shape:", train_labels.shape)
  print("Test Shape:", test_labels.shape)
  print("Type:", train_labels.dtype)
  print("One Example:", train_labels[0])

  #Converts single number labels into a probability distribution
  #Results in float64 labels, which is why the images get casted into the same type
  train_labels = to_categorical(train_labels)
  test_labels = to_categorical(test_labels)

  print("\nLabels After Processing:")
  print("Train Shape:", train_labels.shape)
  print("Test Shape:", test_labels.shape)
  print("Type:", train_labels.dtype)
  print("One Example:", train_labels[0])
  
  print("Image example (first image in training set):")
  
  plt.imshow(train_images[0].reshape(28, 28), cmap='gray')
  plt.title(f"Label: {np.argmax(train_labels[0])}")
  plt.show()

  model = CNN()

  with open(log_file, 'a') as f:
      f.write("\nCNN_00:\n")

  for epoch in range(1, num_epochs+1):
      start_time = time.time()
      train_loss = train_model(model, train_images, train_labels, lr, batch_size)
      test_acc = evaluate_model(model, test_images, test_labels)

      print(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Accuracy: {test_acc} | Time: {time.time() - start_time}")

      with open(log_file, 'a') as f:
          f.write(f"Epoch: {epoch} | Train Loss: {train_loss} | Test Accuracy: {test_acc} | Time: {time.time() - start_time}\n")

