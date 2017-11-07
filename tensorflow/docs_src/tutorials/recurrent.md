# Recurrent Neural Networks

## Introduction

Take a look at [this great article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for an introduction to recurrent neural networks and LSTMs in particular.

## Language Modeling

In this tutorial we will show how to train a recurrent neural network on
a challenging task of language modeling. The goal of the problem is to fit a
probabilistic model which assigns probabilities to sentences. It does so by
predicting next words in a text given a history of previous words. For this
purpose we will use the [Penn Tree Bank](https://catalog.ldc.upenn.edu/ldc99t42)
(PTB) dataset, which is a popular benchmark for measuring the quality of these
models, whilst being small and relatively fast to train.

Language modeling is key to many interesting problems such as speech
recognition, machine translation, or image captioning. It is also fun --
take a look [here](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

For the purpose of this tutorial, we will reproduce the results from
[Zaremba et al., 2014](https://arxiv.org/abs/1409.2329)
([pdf](https://arxiv.org/pdf/1409.2329.pdf)), which achieves very good quality
on the PTB dataset.

## Tutorial Files

This tutorial references the following files from `models/tutorials/rnn/ptb` in the [TensorFlow models repo](https://github.com/tensorflow/models):

File | Purpose
--- | ---
`ptb_word_lm.py` | The code to train a language model on the PTB dataset.
`reader.py` | The code to read the dataset.

## Download and Prepare the Data

The data required for this tutorial is in the `data/` directory of the
[PTB dataset from Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz).

The dataset is already preprocessed and contains overall 10000 different words,
including the end-of-sentence marker and a special symbol (\<unk\>) for rare
words. In `reader.py`, we convert each word to a unique integer identifier,
in order to make it easy for the neural network to process the data.

## The Model

### LSTM

The core of the model consists of an LSTM cell that processes one word at a
time and computes probabilities of the possible values for the next word in the
sentence. The memory state of the network is initialized with a vector of zeros
and gets updated after reading each word. For computational reasons, we will
process data in mini-batches of size `batch_size`.  In this example, it is
important to note that `current_batch_of_words` does not correspond to a
"sentence" of words.  Every word in a batch should correspond to a time t.
TensorFlow will automatically sum the gradients of each batch for you.

For example:
```
 t=0  t=1    t=2  t=3     t=4
[The, brown, fox, is,     quick]
[The, red,   fox, jumped, high]

words_in_dataset[0] = [The, The]
words_in_dataset[1] = [brown, red]
words_in_dataset[2] = [fox, fox]
words_in_dataset[3] = [is, jumped]
words_in_dataset[4] = [quick, high]
batch_size = 2, time_steps = 5
```

The basic pseudocode is as follows:

```python
words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
hidden_state = tf.zeros([batch_size, lstm.state_size])
current_state = tf.zeros([batch_size, lstm.state_size])
state = hidden_state, current_state
probabilities = []
loss = 0.0
for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)
```

### Truncated Backpropagation

By design, the output of a recurrent neural network (RNN) depends on arbitrarily
distant inputs. Unfortunately, this makes backpropagation computation difficult.
In order to make the learning process tractable, it is common practice to create
an "unrolled" version of the network, which contains a fixed number
(`num_steps`) of LSTM inputs and outputs. The model is then trained on this
finite approximation of the RNN. This can be implemented by feeding inputs of
length `num_steps` at a time and performing a backward pass after each
such input block.

Here is a simplified block of code for creating a graph which performs
truncated backpropagation:

```python
# Placeholder for the inputs in a given iteration.
words = tf.placeholder(tf.int32, [batch_size, num_steps])

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
initial_state = state = tf.zeros([batch_size, lstm.state_size])

for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state
```

And this is how to implement an iteration over the whole dataset:

```python
# A numpy array holding the state of LSTM after each batch of words.
numpy_state = initial_state.eval()
total_loss = 0.0
for current_batch_of_words in words_in_dataset:
    numpy_state, current_loss = session.run([final_state, loss],
        # Initialize the LSTM state from the previous iteration.
        feed_dict={initial_state: numpy_state, words: current_batch_of_words})
    total_loss += current_loss
```

### Inputs

The word IDs will be embedded into a dense representation (see the
@{$word2vec$Vector Representations Tutorial}) before feeding to
the LSTM. This allows the model to efficiently represent the knowledge about
particular words. It is also easy to write:

```python
# embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)
```

The embedding matrix will be initialized randomly and the model will learn to
differentiate the meaning of words just by looking at the data.

### Loss Function

We want to minimize the average negative log probability of the target words:

$$ \text{loss} = -\frac{1}{N}\sum_{i=1}^{N} \ln p_{\text{target}_i} $$

It is not very difficult to implement but the function
`sequence_loss_by_example` is already available, so we can just use it here.

The typical measure reported in the papers is average per-word perplexity (often
just called perplexity), which is equal to

$$e^{-\frac{1}{N}\sum_{i=1}^{N} \ln p_{\text{target}_i}} = e^{\text{loss}} $$

and we will monitor its value throughout the training process.

### Stacking multiple LSTMs

To give the model more expressive power, we can add multiple layers of LSTMs
to process the data. The output of the first layer will become the input of
the second and so on.

We have a class called `MultiRNNCell` that makes the implementation seamless:

```python
def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(number_of_layers)])

initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = stacked_lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state
```

## Run the Code

Before running the code, download the PTB dataset, as discussed at the beginning
of this tutorial.  Then, extract the PTB dataset underneath your home directory
as follows:

```bsh
tar xvfz simple-examples.tgz -C $HOME
```
_(Note: On Windows, you may need to use
[other tools](https://wiki.haskell.org/How_to_unpack_a_tar_file_in_Windows).)_

Now, clone the [TensorFlow models repo](https://github.com/tensorflow/models)
from GitHub. Run the following commands:

```bsh
cd models/tutorials/rnn/ptb
python ptb_word_lm.py --data_path=$HOME/simple-examples/data/ --model=small
```

There are 3 supported model configurations in the tutorial code: "small",
"medium" and "large". The difference between them is in size of the LSTMs and
the set of hyperparameters used for training.

The larger the model, the better results it should get. The `small` model should
be able to reach perplexity below 120 on the test set and the `large` one below
80, though it might take several hours to train.

## What Next?

There are several tricks that we haven't mentioned that make the model better,
including:

* decreasing learning rate schedule,
* dropout between the LSTM layers.

Study the code and modify it to improve the model even further.
