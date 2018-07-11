# RevNet with TensorFlow eager execution

This folder contains an TensorFlow eager implementation of the [Reversible Residual Network](https://arxiv.org/pdf/1707.04585.pdf) adapted from the released implementation by the authors. The presented implementation can be ran both in eager and graph mode. The code is considerably simplified with `tf.GradientTape`. Moreover, we reduce the step of reconstructing the outputs. This saves us from using `tf.stop_gradient` and makes the model run faster.

##  Content

- `revnet.py`: The RevNet model.
- `blocks.py`: The relevant reversible blocks.
- `cifar_tfrecords.py`: Script to generate the TFRecords for both CIFAR-10 and CIFAR-100.
- `cifar_input.py`: Script to read from TFRecords and generate dataset objects with the `tf.data` API.
- `config.py`: Configuration file for network architectures and training hyperparameters.
- `main.py`: Main training and evaluation script.
- `ops.py`: Auxiliary downsampling operation.

## To run
- Make sure you have installed TensorFlow 1.9+ or the latest `tf-nightly`
or `tf-nightly-gpu` pip package in order to access the eager execution feature.

- First run

```bash
python cifar_tfrecords.py --data_dir ${PWD}/cifar
```
to download the cifar dataset and convert them
to TFRecords. This produces TFRecord files for both CIFAR-10 and CIFAR-100.

- To train a model run

```bash
python main.py --data_dir ${PWD}/cifar
```

- Optional arguments for `main.py` include
  - `train_dir`: Directory to store eventfiles and checkpoints.
  - `restore`: Restore the latest checkpoint.
  - `validate`: Use validation set for training monitoring.
  - `manual_grad`: Use the manually defined gradient map given by the authors.
  - `dataset`: Use either `cifar-10` or `cifar-100`

## Performance
- With the current implementation, RevNet-38 achieves >92% on CIFAR-10 and >71% on CIFAR-100.

## Reference
The Reversible Residual Network: Backpropagation Without Storing Activations.
Aidan N. Gomez, Mengye Ren, Raquel Urtasun, Roger B. Grosse. Neural Information Processing Systems (NIPS), 2017.
