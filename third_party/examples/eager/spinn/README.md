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

Other eager execution examples can be found under [tensorflow/contrib/eager/python/examples](../../../../tensorflow/contrib/eager/python/examples).

##  Content

- [`data.py`](../../../../tensorflow/contrib/eager/python/examples/spinn/data.py): Pipeline for loading and preprocessing the
   [SNLI](https://nlp.stanford.edu/projects/snli/) data and
   [GloVe](https://nlp.stanford.edu/projects/glove/) word embedding, written
   using the [`tf.data`](https://www.tensorflow.org/programmers_guide/datasets)
   API.
- [`spinn.py`](./spinn.py): Model definition and training routines.
  This example illustrates how one might perform the following actions with
  eager execution enabled:
  * defining a model consisting of a dynamic computation graph,
  * assigning operations to the CPU or GPU dependending on device availability,
  * training the model using the data from the `tf.data`-based pipeline,
  * obtaining metrics such as mean accuracy during training,
  * saving and loading checkpoints,
  * writing summaries for monitoring and visualization in TensorBoard.

## To run

- Make sure you have installed TensorFlow release 1.5 or higher. Alternatively,
  you can use the latest `tf-nightly` or `tf-nightly-gpu` pip
  package to access the eager execution feature.

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

- After training, you may use the model to perform inference on input data in
  the SNLI data format. The premise and hypotheses sentences are specified with
  the command-line flags `--inference_premise` and `--inference_hypothesis`,
  respectively. Each sentence should include the words, as well as parentheses
  representing a binary parsing of the sentence. The words and parentheses
  should all be separated by spaces. For instance,

  ```bash
  python spinn.py --data_root /tmp/spinn-data --logdir /tmp/spinn-logs \
      --inference_premise '( ( The dog ) ( ( is running ) . ) )' \
      --inference_hypothesis '( ( The dog ) ( moves . ) )'
  ```

  which will generate an output like the following, due to the semantic
  consistency of the two sentences.

  ```none
  Inference logits:
    entailment:     1.101249 (winner)
    contradiction:  -2.374171
    neutral:        -0.296733
  ```

  By contrast, the following sentence pair:

  ```bash
  python spinn.py --data_root /tmp/spinn-data --logdir /tmp/spinn-logs \
      --inference_premise '( ( The dog ) ( ( is running ) . ) )' \
      --inference_hypothesis '( ( The dog ) ( rests . ) )'
  ```

  will give you an output like the following, due to the semantic
  contradiction of the two sentences.

  ```none
  Inference logits:
    entailment:     -1.070098
    contradiction:  2.798695 (winner)
    neutral:        -1.402287
  ```
