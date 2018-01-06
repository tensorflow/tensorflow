# GAN with TensorFlow eager execution

A simple Generative Adversarial Network (GAN) example using eager execution.
The discriminator and generator networks each contain a few convolution and
fully connected layers.

Other eager execution examples can be found under the parent directory.

##  Content

- `mnist.py`: Model definitions and training routines.
- `mnist_test.py`: Benchmarks for training and using the models using eager
execution.
- `mnist_graph_test.py`: Benchmarks for trainig and using the models using
graph execution. The same model definitions and loss functions are used in
all benchmarks.


## To run

- Make sure you have installed TensorFlow 1.5+ or the latest `tf-nightly`
or `tf-nightly-gpu` pip package in order to access the eager execution feature.

- Train model. E.g.,

  ```bash
  python mnist.py
  ```
  
  Use `--output_dir=<DIR>` to direct the script to save TensorBoard summaries
  during training. Disabled by default.
  
  Use `--checkpoint_dir=<DIR>` to direct the script to save checkpoints to
  `<DIR>` during training. DIR defaults to /tmp/tensorflow/mnist/checkpoints/.
  The script will load the   latest saved checkpoint from this directory if
  one exists.
  
  Use `-h` for other options.
