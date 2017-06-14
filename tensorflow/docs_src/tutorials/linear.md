# Large-scale Linear Models with TensorFlow

The tf.contrib.learn API provides (among other things) a rich set of tools for working
with linear models in TensorFlow. This document provides an overview of those
tools. It explains:

   * what a linear model is.
   * why you might want to use a linear model.
   * how tf.contrib.learn makes it easy to build linear models in TensorFlow.
   * how you can use tf.contrib.learn to combine linear models with
   deep learning to get the advantages of both.

Read this overview to decide whether the tf.contrib.learn linear model tools might be
useful to you. Then do the @{$wide$Linear Models tutorial} to
give it a try. This overview uses code samples from the tutorial, but the
tutorial walks through the code in greater detail.

To understand this overview it will help to have some familiarity
with basic machine learning concepts, and also with
@{$tflearn$tf.contrib.learn}.

[TOC]

## What is a linear model?

A *linear model* uses a single weighted sum of features to make a prediction.
For example, if you have [data](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)
on age, years of education, and weekly hours of
work for a population, you can learn weights for each of those numbers so that
their weighted sum estimates a person's salary. You can also use linear models
for classification.

Some linear models transform the weighted sum into a more convenient form. For
example, *logistic regression* plugs the weighted sum into the logistic
function to turn the output into a value between 0 and 1. But you still just
have one weight for each input feature.

## Why would you want to use a linear model?

Why would you want to use so simple a model when recent research has
demonstrated the power of more complex neural networks with many layers?

Linear models:

   * train quickly, compared to deep neural nets.
   * can work well on very large feature sets.
   * can be trained with algorithms that don't require a lot of fiddling
   with learning rates, etc.
   * can be interpreted and debugged more easily than neural nets.
   You can examine the weights assigned to each feature to figure out what's
   having the biggest impact on a prediction.
   * provide an excellent starting point for learning about machine learning.
   * are widely used in industry.

## How does tf.contrib.learn help you build linear models?

You can build a linear model from scratch in TensorFlow without the help of a
special API. But tf.contrib.learn provides some tools that make it easier to build
effective large-scale linear models.

### Feature columns and transformations

Much of the work of designing a linear model consists of transforming raw data
into suitable input features. tf.contrib.learn uses the `FeatureColumn` abstraction to
enable these transformations.

A `FeatureColumn` represents a single feature in your data. A `FeatureColumn`
may represent a quantity like 'height', or it may represent a category like
'eye_color' where the value is drawn from a set of discrete possibilities like {'blue', 'brown', 'green'}.

In the case of both *continuous features* like 'height' and *categorical
features* like 'eye_color', a single value in the data might get transformed
into a sequence of numbers before it is input into the model. The
`FeatureColumn` abstraction lets you manipulate the feature as a single
semantic unit in spite of this fact. You can specify transformations and
select features to include without dealing with specific indices in the
tensors you feed into the model.

#### Sparse columns

Categorical features in linear models are typically translated into a sparse
vector in which each possible value has a corresponding index or id. For
example, if there are only three possible eye colors you can represent
'eye_color' as a length 3 vector: 'brown' would become [1, 0, 0], 'blue' would
become [0, 1, 0] and 'green' would become [0, 0, 1]. These vectors are called
"sparse" because they may be very long, with many zeros, when the set of
possible values is very large (such as all English words).

While you don't need to use sparse columns to use tf.contrib.learn linear models, one
of the strengths of linear models is their ability to deal with large sparse
vectors. Sparse features are a primary use case for the tf.contrib.learn linear model
tools.

##### Encoding sparse columns

`FeatureColumn` handles the conversion of categorical values into vectors
automatically, with code like this:

```python
eye_color = tf.contrib.layers.sparse_column_with_keys(
  column_name="eye_color", keys=["blue", "brown", "green"])
```

where `eye_color` is the name of a column in your source data.

You can also generate `FeatureColumn`s for categorical features for which you
don't know all possible values. For this case you would use
`sparse_column_with_hash_bucket()`, which uses a hash function to assign
indices to feature values.

```python
education = tf.contrib.layers.sparse_column_with_hash_bucket(\
    "education", hash_bucket_size=1000)
```

##### Feature Crosses

Because linear models assign independent weights to separate features, they
can't learn the relative importance of specific combinations of feature
values. If you have a feature 'favorite_sport' and a feature 'home_city' and
you're trying to predict whether a person likes to wear red, your linear model
won't be able to learn that baseball fans from St. Louis especially like to
wear red.

You can get around this limitation by creating a new feature
'favorite_sport_x_home_city'. The value of this feature for a given person is
just the concatenation of the values of the two source features:
'baseball_x_stlouis', for example. This sort of combination feature is called
a *feature cross*.

The `crossed_column()` method makes it easy to set up feature crosses:

```python
sport = tf.contrib.layers.sparse_column_with_hash_bucket(\
    "sport", hash_bucket_size=1000)
city = tf.contrib.layers.sparse_column_with_hash_bucket(\
    "city", hash_bucket_size=1000)
sport_x_city = tf.contrib.layers.crossed_column(
    [sport, city], hash_bucket_size=int(1e4))
```

#### Continuous columns

You can specify a continuous feature like so:

```python
age = tf.contrib.layers.real_valued_column("age")
```

Although, as a single real number, a continuous feature can often be input
directly into the model, tf.contrib.learn offers useful transformations for this sort
of column as well.

##### Bucketization

*Bucketization* turns a continuous column into a categorical column. This
transformation lets you use continuous features in feature crosses, or learn
cases where specific value ranges have particular importance.

Bucketization divides the range of possible values into subranges called
buckets:

```python
age_buckets = tf.contrib.layers.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

The bucket into which a value falls becomes the categorical label for
that value.

#### Input function

`FeatureColumn`s provide a specification for the input data for your model,
indicating how to represent and transform the data. But they do not provide
the data itself. You provide the data through an input function.

The input function must return a dictionary of tensors. Each key corresponds to
the name of a `FeatureColumn`. Each key's value is a tensor containing the
values of that feature for all data instances. See
@{$input_fn$Building Input Functions with tf.contrib.learn} for a
more comprehensive look at input functions, and `input_fn` in the
[linear models tutorial code](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)
for an example implementation of an input function.

The input function is passed to the `fit()` and `evaluate()` calls that
initiate training and testing, as described in the next section.

### Linear estimators

tf.contrib.learn's estimator classes provide a unified training and evaluation harness
for regression and classification models. They take care of the details of the
training and evaluation loops and allow the user to focus on model inputs and
architecture.

To build a linear estimator, you can use either the
`tf.contrib.learn.LinearClassifier` estimator or the
`tf.contrib.learn.LinearRegressor` estimator, for classification and
regression respectively.

As with all tf.contrib.learn estimators, to run the estimator you just:

   1. Instantiate the estimator class. For the two linear estimator classes,
   you pass a list of `FeatureColumn`s to the constructor.
   2. Call the estimator's `fit()` method to train it.
   3. Call the estimator's `evaluate()` method to see how it does.

For example:

```python
e = tf.contrib.learn.LinearClassifier(feature_columns=[
  native_country, education, occupation, workclass, marital_status,
  race, age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  model_dir=YOUR_MODEL_DIRECTORY)
e.fit(input_fn=input_fn_train, steps=200)
# Evaluate for one step (one pass through the test data).
results = e.evaluate(input_fn=input_fn_test, steps=1)

# Print the stats for the evaluation.
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
```

### Wide and deep learning

The tf.contrib.learn API also provides an estimator class that lets you jointly train
a linear model and a deep neural network. This novel approach combines the
ability of linear models to "memorize" key features with the generalization
ability of neural nets. Use `tf.contrib.learn.DNNLinearCombinedClassifier` to
create this sort of "wide and deep" model:

```python
e = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=YOUR_MODEL_DIR,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
```
For more information, see the @{$wide_and_deep$Wide and Deep Learning tutorial}.
