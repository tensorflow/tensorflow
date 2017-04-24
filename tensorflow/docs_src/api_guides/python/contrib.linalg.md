# Linear Algebra (contrib)
[TOC]

Linear algebra libraries for TensorFlow.

## `LinearOperator`

Subclasses of `LinearOperator` provide a access to common methods on a
(batch) matrix, without the need to materialize the matrix.  This allows:

* Matrix free computations
* Different operators to take advantage of special strcture, while providing a
  consistent API to users.

### Base class

*   @{tf.contrib.linalg.LinearOperator}

### Individual operators

*   @{tf.contrib.linalg.LinearOperatorDiag}
*   @{tf.contrib.linalg.LinearOperatorIdentity}
*   @{tf.contrib.linalg.LinearOperatorScaledIdentity}
*   @{tf.contrib.linalg.LinearOperatorMatrix}
*   @{tf.contrib.linalg.LinearOperatorTriL}
*   @{tf.contrib.linalg.LinearOperatorUDVHUpdate}

### Transformations and Combinations of operators

*   @{tf.contrib.linalg.LinearOperatorComposition}
