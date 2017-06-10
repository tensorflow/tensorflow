Library of multidimensional LSTM models.

# 2D LSTM code

The [!2D LSTM](https://www.cs.toronto.edu/~graves/nips_2008.pdf) layers take tensors of the form (batch_size, height, width,
depth), compatible with convolutional layers, as inputs. The library
make 2d block from images and process it using information from 2 neighbour
cells.

The library currently provides:

 - a  2D LSTM layer that uses information from both vertical and horizontal neighbour
 - a function that run 2dLSTM in all 4 directions and readuce mean

