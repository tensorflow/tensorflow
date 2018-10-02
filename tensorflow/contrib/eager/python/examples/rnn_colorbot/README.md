RNN Colorbot: An RNN that predicts colors using eager execution.

To train and generate colors, run:

```
python rnn_colorbot.py
```

This example shows how to:
  1. read, process, (one-hot) encode, and pad text data via the
     Datasets API;
  2. build a trainable model;
  3. implement a multi-layer RNN using Python control flow
     constructs (e.g., a for loop);
  4. train a model using an iterative gradient-based method; and
  5. log training and evaluation loss for consumption by TensorBoard
     (to view summaries, use: tensorboard --log_dir=<dir>/summaries).

The data used in this example is licensed under the Creative Commons
Attribution-ShareAlike License and is available at
  https://en.wikipedia.org/wiki/List_of_colors:_A-F
  https://en.wikipedia.org/wiki/List_of_colors:_G-M
  https://en.wikipedia.org/wiki/List_of_colors:_N-Z

This example was adapted from
  https://github.com/random-forests/tensorflow-workshop/tree/master/archive/extras/colorbot
