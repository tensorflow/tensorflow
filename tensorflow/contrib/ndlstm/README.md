Library of multidimensional LSTM models and related code.

# 2D LSTM code

The 2D LSTM layers take tensors of the form (batch_size, height, width,
depth), compatible with convolutional layers, as inputs. The library
transposes and reshapes these tensors in a way that allows batches of
images to be processed by LSTMs.

The library currently provides:

 - a separable 2D LSTM layer
 - a simple 2D convolutional layer that can be swapped out against 2D LSTM
 - layers to reduce images to sequences and images to final state vectors
 - layers for sequence classification, pixel-wise classification

# Other Dimensions

There is 1D LSTM code in `lstm1d.py`. This code implements 1D LSTM versions
suitable as a basis for higher dimensional LSTMs. It is intended for constant
batch size and uses a different layout.  Although the code is perfectly fine for
1D use, you may find other 1D LSTM implementations to be more convenient if you
are interested in sequence problems.

# Upcoming Changes

 - PyramidLSTM
 - support for 3D and 4D
 - optional use of native fused LSTM op
 - easy-to-use command line drivers and examples
 - operators for patch-wise processing
