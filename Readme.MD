## Assignment Outline

- In this assignment I tried to build a CNN model using pytorch.

## How to create a model ?

```
model = SimpleCNN(num_filters, filter_sizes, activation_fn, num_neurons_dense, use_batch_norm, dropout_prob)
```

- details of the parameters are listed below

## How to train the model ?

```
train(model, criterion, optimizer, num_epochs, trainDataLoader, valDataLoader)
```

- train function takes model, optimizer, and no.of epochs and the train and val data loaders.

# Part A

In the part A of the assignment we tried to implement the cnn model we are talking.

train_A.py is the file one needs to run to see the running of the model and it takes follow arguments.

## Command-Line Arguments

Below are the command-line arguments supported by the script, sepcifing default values and the inputs it will take

- `--epochs`, `-e`  
  **Description:** Number of epochs to train the neural network.  
  **Type:** `int`  
  **Default:** `10`

- `--batch_size`, `-b`  
  **Description:** Batch size used to train the neural network.  
  **Type:** `int`  
  **Default:** `16`

- `--optimizer`, `-o`  
  **Description:** Optimizer used for training.  
  **Choices:** `nadam`, `adam`, `rmsprop`  
  **Default:** `nadam`

- `--learning_rate`, `-lr`  
  **Description:** Learning rate for the optimizer.  
  **Type:** `float`  
  **Default:** `0.1`

- `--weight_decay`, `-w_d`  
  **Description:** Weight decay factor for regularization.  
  **Type:** `float`  
  **Default:** `0.0`

- `--activation`, `-a`  
  **Description:** Activation function used in the neural network layers.  
  **Choices:** `relu`, `elu`, `selu`, `silu`, `gelu`, `mish`  
  **Default:** `relu`

- `--num_filters`, `-nf`  
  **Description:** Number of filters for convolutional layers, specified as a list of integers.  
  **Type:** `int[]` (List of integers)  
  **Default:** `[32, 32, 32, 32, 32]`

- `--filter_sizes`, `-fs`  
  **Description:** Sizes of the filters for convolutional layers, specified as a list of integers.  
  **Type:** `int[]` (List of integers)  
  **Default:** `[3, 3, 3, 3, 3]`

- `--batch_norm`, `-bn`  
  **Description:** Indicates whether batch normalization is applied.  
  **Choices:** `true`, `false`  
  **Default:** `true`

- `--dense_layer`, `-dl`  
  **Description:** Number of units in the dense layer.  
  **Type:** `int`  
  **Default:** `128`

- `--augmentation`, `-a`  
  **Description:** Indicates whether data augmentation is applied.  
  **Choices:** `yes`, `no`  
  **Default:** `yes`

- `--dropout`, `-dp`  
  **Description:** Dropout rate for dropout layers.  
  **Type:** `float`  
  **Default:** `0.2`

- `--base_dir`, `-br`  
  **Description:** Base directory for the dataset.  
  **Default:** `inaturalist_12K`  
  **More Info:** Please add the base directory not followed by '\\'

# Part B

In this we are going to use the already existed model here I am using Resnet50.

- Here I am using the Renet50 and training that.
- for this you can pass the learning rate and weight decay and check how much accuracy you are getting.

now to run the model and check its running accuracies

```
config_and_train(base_dir, learning_rate, weight_decay, epochs , batchSize, optimiser_fn)
```

## Command-Line Arguments

Below are the command-line arguments supported by the script, specifying default values and the inputs it will take:

- `--epochs`, `-e`  
  **Description:** Specifies the number of epochs for which the neural network should be trained. An epoch is a full training cycle on the entire dataset.  
  **Type:** `int`  
  **Default:** `10`

- `--batch_size`, `-b`  
  **Description:** Sets the batch size for training the neural network. The batch size is the number of training examples utilized in one iteration.  
  **Type:** `int`  
  **Default:** `16`

- `--optimizer`, `-o`  
  **Description:** Chooses the optimization algorithm used for minimizing the loss function during training. Different optimizers can have different performance characteristics and requirements.  
  **Type:** `string`  
  **Choices:** `nadam`, `adam`, `rmsprop`  
  **Default:** `nadam`

- `--learning_rate`, `-lr`  
  **Description:** Sets the learning rate, which is a scalar used to train a model via gradient descent. The learning rate controls how much to adjust the model in response to the estimated error each time the model weights are updated.  
  **Type:** `float`  
  **Default:** `0.1`

- `--weight_decay`, `-w_d`  
  **Description:** Adds a regularization term to the loss function to prevent the model from overfitting. This parameter controls the strength of the weight decay.  
  **Type:** `float`  
  **Default:** `0.0`

- `--base_dir`, `-br`  
  **Description:** Specifies the base directory where the dataset is located or where the program should look for the dataset. This can be useful for managing datasets located in different paths.  
  **Type:** `string`  
  **Default:** `inaturalist_12K`
