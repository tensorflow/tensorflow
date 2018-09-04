<!-- TODO(acotter): Add usage example of non-convex optimization and stochastic classification. -->

# ConstrainedOptimization (TFCO)

TFCO is a library for optimizing inequality-constrained problems in TensorFlow.
Both the objective function and the constraints are represented as Tensors,
giving users the maximum amount of flexibility in specifying their optimization
problems.

This flexibility makes optimization considerably more difficult: on a non-convex
problem, if one uses the "standard" approach of introducing a Lagrange
multiplier for each constraint, and then jointly maximizing over the Lagrange
multipliers and minimizing over the model parameters, then a stable stationary
point might not even *exist*. Hence, in some cases, oscillation, instead of
convergence, is inevitable.

Thankfully, it turns out that even if, over the course of optimization, no
*particular* iterate does a good job of minimizing the objective while
satisfying the constraints, the *sequence* of iterates, on average, usually
will. This observation suggests the following approach: at training time, we'll
periodically snapshot the model state during optimization; then, at evaluation
time, each time we're given a new example to evaluate, we'll sample one of the
saved snapshots uniformly at random, and apply it to the example. This
*stochastic model* will generally perform well, both with respect to the
objective function, and the constraints.

In fact, we can do better: it's possible to post-process the set of snapshots to
find a distribution over at most $$m+1$$ snapshots, where $$m$$ is the number of
constraints, that will be at least as good (and will usually be much better)
than the (much larger) uniform distribution described above. If you're unable or
unwilling to use a stochastic model at all, then you can instead use a heuristic
to choose the single best snapshot.

For full details, motivation, and theoretical results on the approach taken by
this library, please refer to:

> Cotter, Jiang and Sridharan. "Two-Player Games for Efficient Non-Convex
> Constrained Optimization".
> [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

which will be referred to as [CoJiSr18] throughout the remainder of this
document.

### Proxy Constraints

Imagine that we want to constrain the recall of a binary classifier to be at
least 90%. Since the recall is proportional to the number of true positive
classifications, which itself is a sum of indicator functions, this constraint
is non-differentiable, and therefore cannot be used in a problem that will be
optimized using a (stochastic) gradient-based algorithm.

For this and similar problems, TFCO supports so-called *proxy constraints*,
which are (at least semi-differentiable) approximations of the original
constraints. For example, one could create a proxy recall function by replacing
the indicator functions with sigmoids. During optimization, each proxy
constraint function will be penalized, with the magnitude of the penalty being
chosen to satisfy the corresponding *original* (non-proxy) constraint.

On a problem including proxy constraints&mdash;even a convex problem&mdash;the
Lagrangian approach discussed above isn't guaranteed to work. However, a
different algorithm, based on minimizing *swap regret*, does work. Aside from
this difference, the recommended procedure for optimizing a proxy-constrained
problem remains the same: periodically snapshot the model during optimization,
and then either find the best $$m+1$$-sized distribution, or heuristically
choose the single best snapshot.

## Components

*   [constrained_minimization_problem](https://www.tensorflow.org/code/tensorflow/contrib/constrained_optimization/python/constrained_minimization_problem.py):
    contains the `ConstrainedMinimizationProblem` interface. Your own
    constrained optimization problems should be represented using
    implementations of this interface.

*   [constrained_optimizer](https://www.tensorflow.org/code/tensorflow/contrib/constrained_optimization/python/constrained_optimizer.py):
    contains the `ConstrainedOptimizer` interface, which is similar to (but
    different from) `tf.train.Optimizer`, with the main difference being that
    `ConstrainedOptimizer`s are given `ConstrainedMinimizationProblem`s to
    optimize, and perform constrained optimization.

    *   [external_regret_optimizer](https://www.tensorflow.org/code/tensorflow/contrib/constrained_optimization/python/external_regret_optimizer.py):
        contains the `AdditiveExternalRegretOptimizer` implementation, which is
        a `ConstrainedOptimizer` implementing the Lagrangian approach discussed
        above (with additive updates to the Lagrange multipliers). You should
        use this optimizer for problems *without* proxy constraints. It may also
        work for problems with proxy constraints, but we recommend using a swap
        regret optimizer, instead.

        This optimizer is most similar to Algorithm 3 in Appendix C.3 of
        [CoJiSr18], and is discussed in Section 3. The two differences are that
        it uses proxy constraints (if they're provided) in the update of the
        model parameters, and uses `tf.train.Optimizer`s, instead of SGD, for
        the "inner" updates.

    *   [swap_regret_optimizer](https://www.tensorflow.org/code/tensorflow/contrib/constrained_optimization/python/swap_regret_optimizer.py):
        contains the `AdditiveSwapRegretOptimizer` and
        `MultiplicativeSwapRegretOptimizer` implementations, which are
        `ConstrainedOptimizer`s implementing the swap-regret minimization
        approach mentioned above (with additive or multiplicative updates,
        respectively, to the parameters associated with the
        constraints&mdash;these parameters are not Lagrange multipliers, but
        play a similar role). You should use one of these optimizers (we suggest
        `MultiplicativeSwapRegretOptimizer`) for problems *with* proxy
        constraints.

        The `MultiplicativeSwapRegretOptimizer` is most similar to Algorithm 2
        in Section 4 of [CoJiSr18], with the difference being that it uses
        `tf.train.Optimizer`s, instead of SGD, for the "inner" updates. The
        `AdditiveSwapRegretOptimizer` differs further in that it performs
        additive (instead of multiplicative) updates of the stochastic matrix.

*   [candidates](https://www.tensorflow.org/code/tensorflow/contrib/constrained_optimization/python/candidates.py):
    contains two functions, `find_best_candidate_distribution` and
    `find_best_candidate_index`. Both of these functions are given a set of
    candidate solutions to a constrained optimization problem, from which the
    former finds the best distribution over at most $$m+1$$ candidates, and the
    latter heuristically finds the single best candidate. As discussed above,
    the set of candidates will typically be model snapshots saved periodically
    during optimization. Both of these functions require that scipy be
    installed.

    The `find_best_candidate_distribution` function implements the approach
    described in Lemma 3 of [CoJiSr18], while `find_best_candidate_index`
    implements the heuristic used for hyperparameter search in the experiments
    of Section 5.2.

## Convex Example with Proxy Constraints

This is a simple example of recall-constrained optimization on simulated data:
we will try to find a classifier that minimizes the average hinge loss while
constraining recall to be at least 90%.

We'll start with the required imports&mdash;notice the definition of `tfco`:

```python
import math
import numpy as np
import tensorflow as tf

tfco = tf.contrib.constrained_optimization
```

We'll now create an implementation of the `ConstrainedMinimizationProblem` class
for this problem. The constructor takes three parameters: a Tensor containing
the classification labels (0 or 1) for every training example, another Tensor
containing the model's predictions on every training example (sometimes called
the "logits"), and the lower bound on recall that will be enforced using a
constraint.

This implementation will contain both constraints *and* proxy constraints: the
former represents the constraint that the true recall (defined in terms of the
*number* of true positives) be at least `recall_lower_bound`, while the latter
represents the same constraint, but on a hinge approximation of the recall.

```python
class ExampleProblem(tfco.ConstrainedMinimizationProblem):

  def __init__(self, labels, predictions, recall_lower_bound):
    self._labels = labels
    self._predictions = predictions
    self._recall_lower_bound = recall_lower_bound
    # The number of positively-labeled examples.
    self._positive_count = tf.reduce_sum(self._labels)

  @property
  def objective(self):
    return tf.losses.hinge_loss(labels=self._labels, logits=self._predictions)

  @property
  def constraints(self):
    true_positives = self._labels * tf.to_float(self._predictions > 0)
    true_positive_count = tf.reduce_sum(true_positives)
    recall = true_positive_count / self._positive_count
    # The constraint is (recall >= self._recall_lower_bound), which we convert
    # to (self._recall_lower_bound - recall <= 0) because
    # ConstrainedMinimizationProblems must always provide their constraints in
    # the form (tensor <= 0).
    #
    # The result of this function should be a tensor, with each element being
    # a quantity that is constrained to be nonpositive. We only have one
    # constraint, so we return a one-element tensor.
    return self._recall_lower_bound - recall

  @property
  def proxy_constraints(self):
    # Use 1 - hinge since we're SUBTRACTING recall in the constraint function,
    # and we want the proxy constraint function to be convex.
    true_positives = self._labels * tf.minimum(1.0, self._predictions)
    true_positive_count = tf.reduce_sum(true_positives)
    recall = true_positive_count / self._positive_count
    # Please see the corresponding comment in the constraints property.
    return self._recall_lower_bound - recall
```

We'll now create a simple simulated dataset by sampling 1000 random
10-dimensional feature vectors from a Gaussian, finding their labels using a
random "ground truth" linear model, and then adding noise by randomly flipping
200 labels.

```python
# Create a simulated 10-dimensional training dataset consisting of 1000 labeled
# examples, of which 800 are labeled correctly and 200 are mislabeled.
num_examples = 1000
num_mislabeled_examples = 200
dimension = 10
# We will constrain the recall to be at least 90%.
recall_lower_bound = 0.9

# Create random "ground truth" parameters to a linear model.
ground_truth_weights = np.random.normal(size=dimension) / math.sqrt(dimension)
ground_truth_threshold = 0

# Generate a random set of features for each example.
features = np.random.normal(size=(num_examples, dimension)).astype(
    np.float32) / math.sqrt(dimension)
# Compute the labels from these features given the ground truth linear model.
labels = (np.matmul(features, ground_truth_weights) >
          ground_truth_threshold).astype(np.float32)
# Add noise by randomly flipping num_mislabeled_examples labels.
mislabeled_indices = np.random.choice(
    num_examples, num_mislabeled_examples, replace=False)
labels[mislabeled_indices] = 1 - labels[mislabeled_indices]
```

We're now ready to construct our model, and the corresponding optimization
problem. We'll use a linear model of the form $$f(x) = w^T x - t$$, where $$w$$
is the `weights`, and $$t$$ is the `threshold`. The `problem` variable will hold
an instance of the `ExampleProblem` class we created earlier.

```python
# Create variables containing the model parameters.
weights = tf.Variable(tf.zeros(dimension), dtype=tf.float32, name="weights")
threshold = tf.Variable(0.0, dtype=tf.float32, name="threshold")

# Create the optimization problem.
constant_labels = tf.constant(labels, dtype=tf.float32)
constant_features = tf.constant(features, dtype=tf.float32)
predictions = tf.tensordot(constant_features, weights, axes=(1, 0)) - threshold
problem = ExampleProblem(
    labels=constant_labels,
    predictions=predictions,
    recall_lower_bound=recall_lower_bound,
)
```

We're almost ready to train our model, but first we'll create a couple of
functions to measure its performance. We're interested in two quantities: the
average hinge loss (which we seek to minimize), and the recall (which we
constrain).

```python
def average_hinge_loss(labels, predictions):
  num_examples, = np.shape(labels)
  signed_labels = (labels * 2) - 1
  total_hinge_loss = np.sum(np.maximum(0.0, 1.0 - signed_labels * predictions))
  return total_hinge_loss / num_examples

def recall(labels, predictions):
  positive_count = np.sum(labels)
  true_positives = labels * (predictions > 0)
  true_positive_count = np.sum(true_positives)
  return true_positive_count / positive_count
```

As was mentioned earlier, external regret optimizers suffice for problems
without proxy constraints, but swap regret optimizers are recommended for
problems *with* proxy constraints. Since this problem contains proxy
constraints, we use the `MultiplicativeSwapRegretOptimizer`.

For this problem, the constraint is fairly easy to satisfy, so we can use the
same "inner" optimizer (an `AdagradOptimizer` with a learning rate of 1) for
optimization of both the model parameters (`weights` and `threshold`), and the
internal parameters associated with the constraints (these are the analogues of
the Lagrange multipliers used by the `MultiplicativeSwapRegretOptimizer`). For
more difficult problems, it will often be necessary to use different optimizers,
with different learning rates (presumably found via a hyperparameter search): to
accomplish this, pass *both* the `optimizer` and `constraint_optimizer`
parameters to `MultiplicativeSwapRegretOptimizer`'s constructor.

Since this is a convex problem (both the objective and proxy constraint
functions are convex), we can just take the last iterate. Periodic snapshotting,
and the use of the `find_best_candidate_distribution` or
`find_best_candidate_index` functions, is generally only necessary for
non-convex problems (and even then, it isn't *always* necessary).

```python
with tf.Session() as session:
  optimizer = tfco.MultiplicativeSwapRegretOptimizer(
      optimizer=tf.train.AdagradOptimizer(learning_rate=1.0))
  train_op = optimizer.minimize(problem)

  session.run(tf.global_variables_initializer())
  for ii in xrange(1000):
    session.run(train_op)

  trained_weights, trained_threshold = session.run((weights, threshold))

trained_predictions = np.matmul(features, trained_weights) - trained_threshold
print("Constrained average hinge loss = %f" % average_hinge_loss(
    labels, trained_predictions))
print("Constrained recall = %f" % recall(labels, trained_predictions))
```

Running the above code gives the following output (due to the randomness of the
dataset, you'll get a different result when you run it):

```none
Constrained average hinge loss = 0.710019
Constrained recall = 0.899811
```

As we hoped, the recall is extremely close to 90%&mdash;and, thanks to the use
of proxy constraints, this is the *true* recall, not a hinge approximation.

For comparison, let's try optimizing the same problem *without* the recall
constraint:

```python
with tf.Session() as session:
  optimizer = tf.train.AdagradOptimizer(learning_rate=1.0)
  # For optimizing the unconstrained problem, we just minimize the "objective"
  # portion of the minimization problem.
  train_op = optimizer.minimize(problem.objective)

  session.run(tf.global_variables_initializer())
  for ii in xrange(1000):
    session.run(train_op)

  trained_weights, trained_threshold = session.run((weights, threshold))

trained_predictions = np.matmul(features, trained_weights) - trained_threshold
print("Unconstrained average hinge loss = %f" % average_hinge_loss(
    labels, trained_predictions))
print("Unconstrained recall = %f" % recall(labels, trained_predictions))
```

This code gives the following output (again, you'll get a different answer,
since the dataset is random):

```none
Unconstrained average hinge loss = 0.627271
Unconstrained recall = 0.793951
```

Because there is no constraint, the unconstrained problem does a better job of
minimizing the average hinge loss, but naturally doesn't approach 90% recall.
