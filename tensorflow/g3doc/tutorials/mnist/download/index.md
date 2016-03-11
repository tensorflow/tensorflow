# MNIST Data Download

Code: [tensorflow/examples/tutorials/mnist/](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/)

The goal of this tutorial is to show how to download the dataset files required
for handwritten digit classification using the (classic) MNIST data set.

## Tutorial Files

This tutorial references the following files:

File | Purpose
--- | ---
[`input_data.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/input_data.py) | The code to download the MNIST dataset for training and evaluation.

## Prepare the Data

MNIST is a classic problem in machine learning. The problem is to look at
greyscale 28x28 pixel images of handwritten digits and determine which digit
the image represents, for all the digits from zero to nine.

![MNIST Digits](../../../images/mnist_digits.png "MNIST Digits")

For more information, refer to [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
or [Chris Olah's visualizations of MNIST](http://colah.github.io/posts/2014-10-Visualizing-MNIST/).

### Download

[Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
also hosts the training and test data for download.

File | Purpose
--- | ---
[`train-images-idx3-ubyte.gz`](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) | training set images - 55000 training images, 5000 validation images
[`train-labels-idx1-ubyte.gz`](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz) | training set labels matching the images
[`t10k-images-idx3-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz) | test set images - 10000 images
[`t10k-labels-idx1-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz) | test set labels matching the images

In the `input_data.py` file, the `maybe_download()` function will ensure these
files are downloaded into a local data folder for training.

The folder name is specified in a flag variable at the top of the
`fully_connected_feed.py` file and may be changed to fit your needs.

### Unpack and Reshape

The files themselves are not in any standard image format and are manually
unpacked (following the instructions available at the website) by the
`extract_images()` and `extract_labels()` functions in `input_data.py`.

The image data is extracted into a 2d tensor of: `[image index, pixel index]`
where each entry is the intensity value of a specific pixel in a specific
image, rescaled from `[0, 255]` to `[0, 1]`.  The "image index" corresponds
to an image in the dataset, counting up from zero to the size of the dataset.
And the "pixel index" corresponds to a specific pixel in that image, ranging
from zero to the number of pixels in the image.

The 60000 examples in the `train-*` files are then split into 55000 examples
for training and 5000 examples for validation. For all of the 28x28
pixel greyscale images in the datasets the image size is 784 and so the output
tensor for the training set images is of shape `[55000, 784]`.

The label data is extracted into a 1d tensor of: `[image index]`
with the class identifier for each example as the value. For the training set
labels, this would then be of shape `[55000]`.

### DataSet Object

The underlying code will download, unpack, and reshape images and labels for
the following datasets:

Dataset | Purpose
--- | ---
`data_sets.train` | 55000 images and labels, for primary training.
`data_sets.validation` | 5000 images and labels, for iterative validation of training accuracy.
`data_sets.test` | 10000 images and labels, for final testing of trained accuracy.

The `read_data_sets()` function will return a dictionary with a `DataSet`
instance for each of these three sets of data.  The `DataSet.next_batch()`
method can be used to fetch a tuple consisting of `batch_size` lists of images
and labels to be fed into the running TensorFlow session.

```python
images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
```
