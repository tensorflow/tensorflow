# Broadcasting semantics

This document describes how the broadcasting semantics in XLA work.

## What is broadcasting?

Broadcasting is the process of making arrays with different shapes have
compatible shapes for arithmetic operations. The terminology is borrowed from
Numpy
[(broadcasting)](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

Broadcasting may be required for operations between multi-dimensional arrays of
different ranks, or between multi-dimensional arrays with different but
compatible shapes. Consider the addition `X+v` where `X` is a matrix (an array
of rank 2) and `v` is a vector (an array of rank 1). To perform element-wise
addition, XLA needs to "broadcast" the vector `v` to the same rank as the
matrix `X`, by replicating `v` a certain number of times. The vector's length
has to match at least one of the dimensions of the matrix.

For example:

    |1 2 3| + |7 8 9|
    |4 5 6|

The matrix's dimensions are (2,3), the vector's are (3). The vector is broadcast
by replicating it over rows to get:

    |1 2 3| + |7 8 9| = |8  10 12|
    |4 5 6|   |7 8 9|   |11 13 15|

In Numpy, this is called [broadcasting]
(http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Principles

XLA is a low-level infrastructure with a XLA language this is as strict and
explicit as possible, avoiding implicit and "magical" features that may make
some computations slightly easier to define, at the cost of more assumptions
baked into user code that will be difficult to change in the long term. If
necessary, implicit and magical features can be added in client-level wrappers.

In regards to broadcasting, explicit broadcasting specifications on operations
between arrays of different ranks is required. This is different from Numpy,
which infers the specification when possible.

## Broadcasting a lower-rank array onto a higher-rank array

*Scalars* can always be broadcast over arrays without an explicit specification
of broadcasting dimensions. An element-wise binary operation between a scalar
and an array means applying the operation with the scalar for each element in
the array. For example, adding a scalar to a matrix means producing a matrix
each element of which is a sum of the scalar with the corresponding input
matrix's element.

    |1 2 3| + 7 = |8  9  10|
    |4 5 6|       |11 12 13|

Most broadcasting needs can be captured by using a tuple of dimensions on a
binary operation. When the inputs to the operation have different ranks, this
broadcasting tuple specifies which dimension(s) in the **higher-rank** array to
match with the **lower-rank** array.

Consider the previous example, instead of adding a scalar to a (2,3) matrix, add
a vector of dimension (3) to a matrix of dimensions (2,3). *Without specifying
broadcasting, this operation is invalid.* To correctly request matrix-vector
addition, specify the broadcasting dimension to be (1), meaning the vector's
dimension is matched to dimension 1 of the matrix. In 2D, if dimension 0 is
considered as rows and dimension 1 as columns, this means that each element of
the vector becomes a column of a size matching the number of rows in the matrix:

    |7 8 9| ==> |7 8 9|
                |7 8 9|

As a more complex example, consider adding a 3-element vector (dimension (3)) to
a 3x3 matrix (dimensions (3,3)). There are two ways broadcasting can happen for
this example:

(1) A broadcasting dimension of 1 can be used. Each vector element becomes a
column and the vector is duplicated for each row in the matrix.

    |7 8 9| ==> |7 8 9|
                |7 8 9|
                |7 8 9|

(2) A broadcasting dimension of 0 can be used. Each vector element becomes a row
and the vector is duplicated for each column in the matrix.

     |7| ==> |7 7 7|
     |8|     |8 8 8|
     |9|     |9 9 9|

> Note: when adding a 2x3 matrix to a 3-element vector, a broadcasting dimension
> of 0 is invalid.

The broadcasting dimensions can be a tuple that describes how a smaller rank
shape is broadcast into a larger rank shape. For example, given a 2x3x4 cuboid
and a 3x4 matrix, a broadcasting tuple (1,2) means matching the matrix to
dimensions 1 and 2 of the cuboid.

This type of broadcast is used in the binary ops in `ComputationBuilder`, if the
`broadcast_dimensions` argument is given. For example, see
[ComputationBuilder::Add](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.cc).
In the XLA source code, this type of broadcasting is sometimes called "InDim"
broadcasting.

### Formal definition

The broadcasting attribute allows matching a lower-rank array to a higher-rank
array, by specifying which dimensions of the higher-rank array to match. For
example, for an array with dimensions MxNxPxQ, a vector with dimension T can be
matched as follows:

              MxNxPxQ

    dim 3:          T
    dim 2:        T
    dim 1:      T
    dim 0:    T

In each case, T has to be equal to the matching dimension of the higher-rank
array. The vector's values are then broadcast from the matched dimension to all
the other dimensions.

To match a TxV matrix onto the MxNxPxQ array, a pair of broadcasting dimensions
are used:

              MxNxPxQ
    dim 2,3:      T V
    dim 1,2:    T V
    dim 0,3:  T     V
    etc...

The order of dimensions in the broadcasting tuple has to be the order in which
the lower-rank array's dimensions are expected to match the higher-rank array's
dimensions. The first element in the tuple says which dimension in the
higher-rank array has to match dimension 0 in the lower-rank array. The second
element for dimension 1, and so on. The order of broadcast dimensions has to be
strictly increasing. For example, in the previous example it is illegal to match
V to N and T to P; it is also illegal to match V to both P and N.

## Broadcasting similar-rank arrays with degenerate dimensions

A related broadcasting problem is broadcasting two arrays that have the same
rank but different dimension sizes. Similarly to Numpy's rules, this is only
possible when the arrays are *compatible*. Two arrays are compatible when all
their dimensions are compatible. Two dimensions are compatible if:

*   They are equal, or
*   One of them is 1 (a "degenerate" dimension)

When two compatible arrays are encountered, the result shape has the maximum
among the two inputs at every dimension index.

Examples:

1.  (2,1) and (2,3) broadcast to (2,3).
2.  (1,2,5) and (7,2,5) broadcast to (7,2,5)
3.  (7,2,5) and (7,1,5) broadcast to (7,2,5)
4.  (7,2,5) and (7,2,6) are incompatible and cannot be broadcast.

A special case arises, and is also supported, where each of the input arrays has
a degenerate dimension at a different index. In this case, the result is an
"outer operation": (2,1) and (1,3) broadcast to (2,3). For more examples,
consult the [Numpy documentation on
broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

## Broadcast composition

Broadcasting of a lower-rank array to a higher-rank array **and** broadcasting
using degenerate dimensions can both be performed in the same binary operation.
For example, a vector of size 4 and an matrix of size 1x2 can be added together
using broadcast dimensions value of (0):

    |1 2 3 4| + [5 6]    // [5 6] is a 1x2 matrix, not a vector.

First the vector is broadcast up to rank 2 (matrix) using the broadcast
dimensions. The single value (0) in the broadcast dimensions indicates that
dimension zero of the vector matches to dimension zero of the matrix. This
produces an matrix of size 4xM where the value M is chosen to match the
corresponding dimension size in the 1x2 array. Therefore, a 4x2 matrix is
produced:

    |1 1| + [5 6]
    |2 2|
    |3 3|
    |4 4|

Then "degenerate dimension broadcasting" broadcasts dimension zero of the 1x2
matrix to match the corresponding dimension size of the right hand side:

    |1 1| + |5 6|     |6  7|
    |2 2| + |5 6|  =  |7  8|
    |3 3| + |5 6|     |8  9|
    |4 4| + |5 6|     |9 10|

A more complicated example is a matrix of size 1x2 added to an array of size
4x3x1 using broadcast dimensions of (1, 2). First the 1x2 matrix is broadcast up
to rank 3 using the broadcast dimensions to produces an intermediate Mx1x2 array
where the dimension size M is determined by the size of the larger operand (the
4x3x1 array) producing a 4x1x2 intermediate array. The M is at dimension 0
(left-most dimension) because the dimensions 1 and 2 are mapped to the
dimensions of the original 1x2 matrix as the broadcast dimension are (1, 2).
This intermediate array can be added to the 4x3x1 matrix using broadcasting of
degenerate dimensions to produce a 4x3x2 array result.
