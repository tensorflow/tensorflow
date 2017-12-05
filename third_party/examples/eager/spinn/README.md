# SPINN with TensorFlow eager execution

SPINN, or Stack-Augmented Parser-Interpreter Neural Network, is a recursive
neural network that utilizes syntactic parse information for natural language
understanding.

SPINN was originally described by:
Bowman, S.R., Gauthier, J., Rastogi A., Gupta, R., Manning, C.D., & Potts, C.
  (2016). A Fast Unified Model for Parsing and Sentence Understanding.
  https://arxiv.org/abs/1603.06021

Our implementation is based on @jekbradbury's PyTorch implementation at:
https://github.com/jekbradbury/examples/blob/spinn/snli/spinn.py,

which was released under the BSD 3-Clause License at:
https://github.com/jekbradbury/examples/blob/spinn/LICENSE

##  Content

Python source file(s):
- `spinn.py`: Model definition and training routines written with TensorFlow
  eager execution idioms.

## To run

- Make sure you have installed the latest `tf-nightly` or `tf-nightly-gpu` pip
  package of TensorFlow in order to access the eager execution feature.

- Download and extract the raw SNLI data and GloVe embedding vectors.
  For example:

  ```bash
  curl -fSsL https://nlp.stanford.edu/projects/snli/snli_1.0.zip --create-dirs -o /tmp/spinn-data/snli/snli_1.0.zip
  unzip -d /tmp/spinn-data/snli /tmp/spinn-data/snli/snli_1.0.zip
  curl -fSsL http://nlp.stanford.edu/data/glove.42B.300d.zip --create-dirs -o /tmp/spinn-data/glove/glove.42B.300d.zip
  unzip -d /tmp/spinn-data/glove /tmp/spinn-data/glove/glove.42B.300d.zip
  ```

- Train model. E.g.,

  ```bash
  python spinn.py --data_root /tmp/spinn-data --logdir /tmp/spinn-logs
  ```

  During training, model checkpoints and TensorBoard summaries will be written
  periodically to the directory specified with the `--logdir` flag.
  The training script will reload a saved checkpoint from the directory if it
  can find one there.

  To view the summaries with TensorBoard:

  ```bash
  tensorboard --logdir /tmp/spinn-logs
  ```
