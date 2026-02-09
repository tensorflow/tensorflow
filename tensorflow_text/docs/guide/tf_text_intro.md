# Introduction to TensorFlow Text

TensorFlow Text provides a collection of text related classes and ops ready to
use with TensorFlow 2.0. The library can perform the preprocessing regularly
required by text-based models, and includes other features useful for sequence
modeling not provided by core TensorFlow.

The benefit of using these ops in your text preprocessing is that they are done
in the TensorFlow graph. You do not need to worry about tokenization in training
being different than the tokenization at inference, or managing preprocessing
scripts.

## Install TensorFlow Text

### Install using pip

When installing TF Text with pip install, note the version of TensorFlow you are
running, as you should specify the corresponding version of TF Text.

```python
pip install -U tensorflow-text==<version>
```

### Build from source

TensorFlow Text must be built in the same environment as TensorFlow. Thus, if
you manually build TF Text, it is highly recommended that you also build
TensorFlow.

If building on MacOS, you must have coreutils installed. It is probably easiest
to do with Homebrew. First, build TensorFlow
[from source](https://www.tensorflow.org/install/source).

Clone the TF Text repo.

```shell
git clone  https://github.com/tensorflow/text.git
```

Finally, run the build script to create a pip package.

```shell
./oss_scripts/run_build.sh
```
