# TensorForest

TensorForest is an implementation of random forests in TensorFlow using an
online, [extremely randomized trees](
https://en.wikipedia.org/wiki/Random_forest#ExtraTrees)
training algorithm.  It supports both
classification (binary and multiclass) and regression (scalar and vector).

## Usage

TensorForest is a tf.learn Estimator:

```import tensorflow as tf

params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
  num_classes=2, num_features=10, regression=False,
  num_trees=50, max_nodes=1000)

classifier =
tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)

classifier.fit(x=x_train, y=y_train)

y_out = classifier.predict(x=x_test)
```

TensorForest users are implored to properly shuffle their training data,
as our training algorithm strongly assumes it is in random order.

## Algorithm

Each tree in the forest is trained independently in parallel.  For each
tree, we maintain the following data:

* The tree structure, giving the two children of each non-leaf node and
the *split* used to route data between them.  Each split looks at a single
input feature and compares it to a threshold value.

* Leaf statistics.  Each leaf needs to gather statistics, and those
statistics have the property that at the end of training, they can be
turned into predictions.  For classification problems, the statistics are
class counts, and for regression problems they are the vector sum of the
values seen at the leaf, along with a count of those values.

* Growing statistics.  Each leaf needs to gather data that will potentially
allow it to grow into a non-leaf parent node.  That data usually consists
of a list of potential splits, along with statistics for each of those splits.
Split statistics in turn consist of leaf statistics for their left and
right branches, along with some other information that allows us to assess
the quality of the split.  For classification problems, that's usually
the [gini
impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)
of the split, while for regression problems it's the mean-squared error.

At the start of training, the tree structure is initialized to a root node,
and the leaf and growing statistics for it are both empty.  Then, for
each batch `{(x_i, y_i)}`  of training data, the following steps are performed:

1. Given the current tree structure, each `x_i` is used to find the leaf
assignment `l_i`.

2. `y_i` is used to update the leaf statistics of leaf `l_i`.

3. If the growing statistics for the leaf `l_i` do not yet contain
`num_splits_to_consider` splits, `x_i` is used to generate another split.
Specifically, a random feature value is chosen, and `x_i`'s value at that
feature is used for the split's threshold.

4. Otherwise, `(x_i, y_i)` is used to update the statistics of every
split in the growing statistics of leaf `l_i`.  If leaf `l_i` has now seen
`split_after_samples` data points since creating all of its potential splits,
the split with the best score is chosen, and the tree structure is grown.

## Parameters

The following ForestHParams parameters are required:

* `num_classes`.  The number of classes in a classification problem, or
the number of dimensions in the output of a regression problem.

* `num_features`.  The number of input features.

The following ForestHParams parameters are important but not required:

* `regression`.  True for regression problems, False for classification tasks.
  Defaults to False (classification).
For regression problems, TensorForests's output are the predicted regression
values.  For classification, the outputs are the per-class probabilities.

* `num_trees`.  The number of trees to create.  Defaults to 100.  There
usually isn't any accuracy gain from using higher values.

* `max_nodes`.  Defaults to 10,000.  No tree is allowed to grow beyond
`max_nodes` nodes, and training stops when all trees in the forest are this
large.

The remaining ForestHParams parameters don't usually require being set by the
user:

* `num_splits_to_consider`.  Defaults to `sqrt(num_features)` capped to be
between 10 and 1000.  In the extremely randomized tree training algorithm,
only this many potential splits are evaluated for each tree node.

* `split_after_samples`.  Defaults to 250.  In our online version of
extremely randomized tree training, we pick a split for a node after it has
accumulated this many training samples.

* `bagging_fraction`.  If less than 1.0,
then each tree sees only a different, random sampled (without replacement),
`bagging_fraction` sized subset of
the training data.  Defaults to 1.0 (no bagging) because it fails to give
any accuracy improvement our experiments so far.

* `feature_bagging_fraction`.  If less than 1.0, then each tree sees only
a different `feature_bagging_fraction * num_features` sized subset of the
input features.  Defaults to 1.0 (no feature bagging).

* `base_random_seed`.  By default (`base_random_seed = 0`), the random number
generator for each tree is seeded by the current time (in microseconds) when
each tree is first created.  Using a non-zero value causes tree training to
be deterministic, in that the i-th tree's random number generator is seeded
with the value `base_random_seed + i`.

## Implementation

The python code in `python/tensor_forest.py` assigns default values to the
parameters, handles both instance and feature bagging, and creates the
TensorFlow graphs for training and inference.  The graphs themselves are
quite simple, as most of the work is done in custom ops.  There is a single
op (`model_ops.tree_predictions_v4`) that does inference for a single tree,
and four custom ops that do training on a single tree over a single batch,
with each op roughly corresponding to one of the four steps from the
algorithm section above.

The training data itself is stored in TensorFlow _resources_, which provide
a means of non-tensor based persistence storage.  (See
`core/framework/resource_mgr.h` for more information about resources.)
The tree
structure is stored in the `DecisionTreeResource` defined in
`kernels/v4/decision-tree-resource.h` and the leaf and growing statistics
are stored in the `FertileStatsResource` defined in
`kernels/v4/fertile-stats-resource.h`.

## More information

* [Kaggle kernel demonstrating TensorForest on Iris
  dataset](https://www.kaggle.com/thomascolthurst/tensorforest-on-iris/notebook)
* [TensorForest
  paper from NIPS 2016 Workshop](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxtbHN5c25pcHMyMDE2fGd4OjFlNTRiOWU2OGM2YzA4MjE)
