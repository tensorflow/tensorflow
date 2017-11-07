# TensorFlow Wide & Deep Learning Tutorial

In the previous @{$wide$TensorFlow Linear Model Tutorial}, we trained a logistic
regression model to predict the probability that the individual has an annual
income of over 50,000 dollars using the
[Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income).
TensorFlow is great for training deep neural networks too, and you might be
thinking which one you should choose—well, why not both? Would it be possible to
combine the strengths of both in one model?

In this tutorial, we'll introduce how to use the tf.estimator API to jointly
train a wide linear model and a deep feed-forward neural network. This approach
combines the strengths of memorization and generalization. It's useful for
generic large-scale regression and classification problems with sparse input
features (e.g., categorical features with a large number of possible feature
values). If you're interested in learning more about how Wide & Deep Learning
works, please check out our [research paper](https://arxiv.org/abs/1606.07792).

![Wide & Deep Spectrum of Models](https://www.tensorflow.org/images/wide_n_deep.svg "Wide & Deep")

The figure above shows a comparison of a wide model (logistic regression with
sparse features and transformations), a deep model (feed-forward neural network
with an embedding layer and several hidden layers), and a Wide & Deep model
(joint training of both). At a high level, there are only 3 steps to configure a
wide, deep, or Wide & Deep model using the tf.estimator API:

1.  Select features for the wide part: Choose the sparse base columns and
    crossed columns you want to use.
1.  Select features for the deep part: Choose the continuous columns, the
    embedding dimension for each categorical column, and the hidden layer sizes.
1.  Put them all together in a Wide & Deep model
    (`DNNLinearCombinedClassifier`).

And that's it! Let's go through a simple example.

## Setup

To try the code for this tutorial:

1.  @{$install$Install TensorFlow} if you haven't already.

2.  Download [the tutorial code](https://github.com/tensorflow/models/tree/master/official/wide_deep/).

3. Execute the data download script we provide to you:

        $ python data_download.py

4. Execute the tutorial code with the following command to train the wide and
deep model described in this tutorial:

        $ python wide_deep.py

Read on to find out how this code builds its model.


## Define Base Feature Columns

First, let's define the base categorical and continuous feature columns that
we'll use. These base columns will be the building blocks used by both the wide
part and the deep part of the model.

```python
import tensorflow as tf

# Continuous columns
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

# To show an example of hashing:
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)

# Transformations.
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

## The Wide Model: Linear Model with Crossed Feature Columns

The wide model is a linear model with a wide set of sparse and crossed feature
columns:

```python
base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]
```

You can also see the @{$wide$TensorFlow Linear Model Tutorial} for more details.

Wide models with crossed feature columns can memorize sparse interactions
between features effectively. That being said, one limitation of crossed feature
columns is that they do not generalize to feature combinations that have not
appeared in the training data. Let's add a deep model with embeddings to fix
that.

## The Deep Model: Neural Network with Embeddings

The deep model is a feed-forward neural network, as shown in the previous
figure. Each of the sparse, high-dimensional categorical features are first
converted into a low-dimensional and dense real-valued vector, often referred to
as an embedding vector. These low-dimensional dense embedding vectors are
concatenated with the continuous features, and then fed into the hidden layers
of a neural network in the forward pass. The embedding values are initialized
randomly, and are trained along with all other model parameters to minimize the
training loss. If you're interested in learning more about embeddings, check out
the TensorFlow tutorial on @{$word2vec$Vector Representations of Words} or
[Word embedding](https://en.wikipedia.org/wiki/Word_embedding) on Wikipedia.

Another way to represent categorical columns to feed into a neural network is
via a one-hot or multi-hot representation. This is often appropriate for
categorical columns with only a few possible values. As an example of a one-hot
representation, for the relationship column, `"Husband"` can be represented as
[1, 0, 0, 0, 0, 0], and `"Not-in-family"` as [0, 1, 0, 0, 0, 0], etc. This is a
fixed representation, whereas embeddings are more flexible and calculated at
training time.

We'll configure the embeddings for the categorical columns using
`embedding_column`, and concatenate them with the continuous columns.
We also use `indicator_column` to create multi-hot representations of some
categorical columns.

```python
deep_columns = [
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(marital_status),
    tf.feature_column.indicator_column(relationship),
    # To show an example of embedding
    tf.feature_column.embedding_column(occupation, dimension=8),
]
```

The higher the `dimension` of the embedding is, the more degrees of freedom the
model will have to learn the representations of the features. For simplicity, we
set the dimension to 8 for all feature columns here. Empirically, a more
informed decision for the number of dimensions is to start with a value on the
order of \\(\log_2(n)\\) or \\(k\sqrt[4]n\\), where \\(n\\) is the number of
unique features in a feature column and \\(k\\) is a small constant (usually
smaller than 10).

Through dense embeddings, deep models can generalize better and make predictions
on feature pairs that were previously unseen in the training data. However, it
is difficult to learn effective low-dimensional representations for feature
columns when the underlying interaction matrix between two feature columns is
sparse and high-rank. In such cases, the interaction between most feature pairs
should be zero except a few, but dense embeddings will lead to nonzero
predictions for all feature pairs, and thus can over-generalize. On the other
hand, linear models with crossed features can memorize these “exception rules”
effectively with fewer model parameters.

Now, let's see how to jointly train wide and deep models and allow them to
complement each other’s strengths and weaknesses.

## Combining Wide and Deep Models into One

The wide models and deep models are combined by summing up their final output
log odds as the prediction, then feeding the prediction to a logistic loss
function. All the graph definition and variable allocations have already been
handled for you under the hood, so you simply need to create a
`DNNLinearCombinedClassifier`:

```python
model = tf.estimator.DNNLinearCombinedClassifier(
            model_dir='/tmp/census_model',
            linear_feature_columns=base_columns + crossed_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50])
```

## Training and Evaluating The Model

Before we train the model, let's read in the Census dataset as we did in the
@{$wide$TensorFlow Linear Model tutorial}. See `data_download.py` as well as
`input_fn` within
[`wide_deep.py`](https://github.com/tensorflow/models/tree/master/official/wide_deep/wide_deep.py).

After reading in the data, you can train and evaluate the model:

```python
# Set up input function generators for the train and test data files.
train_input_fn = input_fn(
    data_file=FLAGS.train_data,
    num_epochs=FLAGS.epochs_per_eval,
    shuffle=True,
    batch_size=FLAGS.batch_size)
eval_input_fn = input_fn(
    data_file=FLAGS.test_data,
    num_epochs=1,
    shuffle=False,
    batch_size=FLAGS.batch_size)

# Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
  model.train(input_fn=train_input_fn)
  results = model.evaluate(input_fn=eval_input_fn)

  # Display evaluation metrics
  print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
  print('-' * 30)
  for key in sorted(results):
    print('%s: %s' % (key, results[key]))
```

The final output accuracy should be somewhere around 85.5%. If you'd like to
see a working end-to-end example, you can download our
[example code](https://github.com/tensorflow/models/tree/master/official/wide_deep/wide_deep.py).

Note that this tutorial is just a quick example on a small dataset to get you
familiar with the API. Wide & Deep Learning will be even more powerful if you
try it on a large dataset with many sparse feature columns that have a large
number of possible feature values. Again, feel free to take a look at our
[research paper](https://arxiv.org/abs/1606.07792) for more ideas about how to
apply Wide & Deep Learning in real-world large-scale machine learning problems.
