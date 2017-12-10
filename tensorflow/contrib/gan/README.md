<!-- TODO(joelshor): Add images to the examples. -->
# TensorFlow-GAN (TFGAN)

TFGAN is a lightweight library for training and evaluating Generative
Adversarial Networks (GANs). This technique allows you to train a network
(called the 'generator') to sample from a distribution, without having to
explicitly model the distribution and without writing an explicit loss. For
example, the generator could learn to draw samples from the distribution of
natural images. For more details on this technique, see
['Generative Adversarial Networks'](https://arxiv.org/abs/1406.2661) by
Goodfellow et al. See [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/gan/) for examples, and [this tutorial](https://github.com/tensorflow/models/tree/master/research/gan/tutorial.ipynb) for an
introduction.

#### Usage
```python
import tensorflow as tf
tfgan = tf.contrib.gan
```

## Why TFGAN?

* Easily train generator and discriminator networks with well-tested, flexible [library calls](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/train.py). You can
mix TFGAN, native TF, and other custom frameworks
* Use already implemented [GAN losses and penalties](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/losses/python/losses_impl.py) (ex Wasserstein loss, gradient penalty, mutual information penalty, etc)
* [Monitor and visualize](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/eval/python/summaries_impl.py) GAN progress during training, and [evaluate](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py) them
* Use already-implemented [tricks](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/features/python/) to stabilize and improve training
* Develop based on examples of [common GAN setups](https://github.com/tensorflow/models/tree/master/research/gan/)
* Use the TFGAN-backed [GANEstimator](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/estimator/python/gan_estimator_impl.py) to easily train a GAN model
* Improvements in TFGAN infrastructure will automatically benefit your TFGAN project
* Stay up-to-date with research as we add more algorithms

## What are the TFGAN components?

TFGAN is composed of several parts which were design to exist independently.
These include the following main pieces (explained in detail below).

* [core](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/train.py):
provides the main infrastructure needed to train a GAN. Training occurs in four phases, and each phase
can be completed by custom-code or by using a TFGAN library call.

* [features](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/features/python/):
Many common GAN operations and normalization techniques are implemented for you
to use, such as instance normalization and conditioning.

* [losses](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/losses/python/):
Easily experiment with already-implemented and well-tested losses and penalties,
such as the Wasserstein loss, gradient penalty, mutual information penalty, etc

* [evaluation](https://www.tensorflow.org/code/tensorflow/contrib/gan/python/eval/python/):
Use `Inception Score` or `Frechet Distance` with a pretrained Inception
network to evaluate your unconditional generative model. You can also use
your own pretrained classifier for more specific performance numbers, or use
other methods for evaluating conditional generative models.

* [examples](https://github.com/tensorflow/models/tree/master/research/gan/) and [tutorial](https://github.com/tensorflow/models/tree/master/research/gan/tutorial.ipynb):
See examples of how to use TFGAN to make GAN training easier, or use the more complicated examples to jumpstart your
own project. These include unconditional and conditional GANs, InfoGANs,
adversarial losses on existing networks, and image-to-image translation.

## Training a GAN model

Training in TFGAN typically consists of the following steps:

1. Specify the input to your networks.
1. Set up your generator and discriminator using a `GANModel`.
1. Specify your loss using a `GANLoss`.
1. Create your train ops using a `GANTrainOps`.
1. Run your train ops.

At each stage, you can either use TFGAN's convenience functions, or you can
perform the step manually for fine-grained control. We provide examples below.

There are various types of GAN setups. For instance, you can train a generator
to sample unconditionally from a learned distribution, or you can condition on
extra information such as a class label. TFGAN is compatible with many setups,
and we demonstrate a few below:

### Examples

#### Unconditional MNIST generation

This example trains a generator to produce handwritten MNIST digits. The generator maps
random draws from a multivariate normal distribution to MNIST digit images. See
['Generative Adversarial Networks'](https://arxiv.org/abs/1406.2661) by
Goodfellow et al.

```python
# Set up the input.
images = mnist_data_provider.provide_data(FLAGS.batch_size)
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# Build the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=mnist.unconditional_generator,  # you define
    discriminator_fn=mnist.unconditional_discriminator,  # you define
    real_data=images,
    generator_inputs=noise)

# Build the GAN loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan_losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan_losses.wasserstein_discriminator_loss)

# Create the train ops, which calculate gradients and apply updates to weights.
train_ops = tfgan.gan_train_ops(
    gan_model,
    gan_loss,
    generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5))

# Run the train ops in the alternating training scheme.
tfgan.gan_train(
    train_ops,
    hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps)],
    logdir=FLAGS.train_log_dir)
```

#### Conditional MNIST generation
This example trains a generator to generate MNIST images *of a given class*.
The generator maps random draws from a multivariate normal distribution and a
one-hot label of the desired digit class to an MNIST digit image. See
['Conditional Generative Adversarial Nets'](https://arxiv.org/abs/1411.1784) by
Mirza and Osindero.

```python
# Set up the input.
images, one_hot_labels = mnist_data_provider.provide_data(FLAGS.batch_size)
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# Build the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=mnist.conditional_generator,  # you define
    discriminator_fn=mnist.conditional_discriminator,  # you define
    real_data=images,
    generator_inputs=(noise, one_hot_labels))

# The rest is the same as in the unconditional case.
...
```
#### Adversarial loss
This example combines an L1 pixel loss and an adversarial loss to learn to
autoencode images. The bottleneck layer can be used to transmit compressed
representations of the image. Neutral networks with pixel-wise loss only tend to
produce blurry results, so the GAN can be used to make the reconstructions more
plausible. See ['Full Resolution Image Compression with Recurrent Neural Networks'](https://arxiv.org/abs/1608.05148) by Toderici et al
for an example of neural networks used for image compression, and ['Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network'](https://arxiv.org/abs/1609.04802) by Ledig et al for a more detailed description of
how GANs can sharpen image output.

```python
# Set up the input pipeline.
images = image_provider.provide_data(FLAGS.batch_size)

# Build the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=nets.autoencoder,  # you define
    discriminator_fn=nets.discriminator,  # you define
    real_data=images,
    generator_inputs=images)

# Build the GAN loss and standard pixel loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan_losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan_losses.wasserstein_discriminator_loss,
    gradient_penalty=1.0)
l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

# Modify the loss tuple to include the pixel loss.
gan_loss = tfgan.losses.combine_adversarial_loss(
    gan_loss, gan_model, l1_pixel_loss, weight_factor=FLAGS.weight_factor)

# The rest is the same as in the unconditional case.
...
```

#### Image-to-image translation
This example maps images in one domain to images of the same size in a different
dimension. For example, it can map segmentation masks to street images, or
grayscale images to color. See ['Image-to-Image Translation with Conditional Adversarial Networks'](https://arxiv.org/abs/1611.07004) by Isola et al for more details.

```python
# Set up the input pipeline.
input_image, target_image = data_provider.provide_data(FLAGS.batch_size)

# Build the generator and discriminator.
gan_model = tfgan.gan_model(
    generator_fn=nets.generator,  # you define
    discriminator_fn=nets.discriminator,  # you define
    real_data=target_image,
    generator_inputs=input_image)

# Build the GAN loss and standard pixel loss.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan_losses.least_squares_generator_loss,
    discriminator_loss_fn=tfgan_losses.least_squares_discriminator_loss)
l1_pixel_loss = tf.norm(gan_model.real_data - gan_model.generated_data, ord=1)

# Modify the loss tuple to include the pixel loss.
gan_loss = tfgan.losses.combine_adversarial_loss(
    gan_loss, gan_model, l1_pixel_loss, weight_factor=FLAGS.weight_factor)

# The rest is the same as in the unconditional case.
...
```

#### InfoGAN
Train a generator to generate specific MNIST digit images, and control for digit style *without using any labels*. See ['InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets'](https://arxiv.org/abs/1606.03657) for more details.

```python
# Set up the input pipeline.
images = mnist_data_provider.provide_data(FLAGS.batch_size)

# Build the generator and discriminator.
gan_model = tfgan.infogan_model(
    generator_fn=mnist.infogan_generator,  # you define
    discriminator_fn=mnist.infogran_discriminator,  # you define
    real_data=images,
    unstructured_generator_inputs=unstructured_inputs,  # you define
    structured_generator_inputs=structured_inputs)  # you define

# Build the GAN loss with mutual information penalty.
gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan_losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan_losses.wasserstein_discriminator_loss,
    gradient_penalty=1.0,
    mutual_information_penalty_weight=1.0)

# The rest is the same as in the unconditional case.
...
```

#### Custom model creation
Train an unconditional GAN to generate MNIST digits, but manually construct
the `GANModel` tuple for more fine-grained control.

```python
# Set up the input pipeline.
images = mnist_data_provider.provide_data(FLAGS.batch_size)
noise = tf.random_normal([FLAGS.batch_size, FLAGS.noise_dims])

# Manually build the generator and discriminator.
with tf.variable_scope('Generator') as gen_scope:
  generated_images = generator_fn(noise)
with tf.variable_scope('Discriminator') as dis_scope:
  discriminator_gen_outputs = discriminator_fn(generated_images)
with variable_scope.variable_scope(dis_scope, reuse=True):
  discriminator_real_outputs = discriminator_fn(images)
generator_variables = variables_lib.get_trainable_variables(gen_scope)
discriminator_variables = variables_lib.get_trainable_variables(dis_scope)
# Depending on what TFGAN features you use, you don't always need to supply
# every `GANModel` field. At a minimum, you need to include the discriminator
# outputs and variables if you want to use TFGAN to construct losses.
gan_model = tfgan.GANModel(
    generator_inputs,
    generated_data,
    generator_variables,
    gen_scope,
    generator_fn,
    real_data,
    discriminator_real_outputs,
    discriminator_gen_outputs,
    discriminator_variables,
    dis_scope,
    discriminator_fn)

# The rest is the same as the unconditional case.
...
```


## Authors
Joel Shor (github: [joel-shor](https://github.com/joel-shor)) and Sergio Guadarrama (github: [sguada](https://github.com/sguada))
