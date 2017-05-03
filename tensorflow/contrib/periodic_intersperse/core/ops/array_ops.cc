#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("PeriodicIntersperse")
    .Attr("T: numbertype")
    .Attr("S: {int32, int64}")
    .Input("values: T")
    .Input("desired_shape: S")
    .Output("output: T")
    .Doc(R"doc(
Periodicly intersperses elements of a tensor to conform to `desired shape`.

This function implements a slightly more generic version of the subpixel
convolutions found in this [paper](https://arxiv.org/abs/1609.05158).

The formula for computing the elements in the `output` tensor is as follows:
  `T` = `values` tensor of rank `R`
  `S` = `desired_shape` tensor (vector of length `R`)
  `P` = `output` tensor of rank `R`
  \((T_1,\ldots,T_R)\) = shape(`T`)
  \([S_1,\ldots,S_q,\ldots,S_R]\) = elements of vector `S`

  A single element in `S` is left unspecified (denoted \(S_q=-1\)).
  Let \(f_i\) denote the (possibly non-integer) factor that relates the original
  dimension to the desired dimensions, \(S_i=f_i T_i\), for \(i\neq q\) where
  \(f_i>0\).
  Define the following:
    \(g_i=\lceil f_i\rceil\)
    \(t=\prod_i T_i\)
    \(s=\prod_{i\neq q} S_i\)
  \(S_q\) can then be defined as by \(S_q=\lfloor t/s\rfloor\).
  The elements of the resulting tensor are defined as
  \(P_{s_1,\ldots,s_R}=T_{h_1,\ldots,h_q,\ldots,h_R}\).
  The \(h_i\) (\(i\neq q\)) are defined by \(h_i=\lfloor s_i/g_i\rfloor\).
  \(h_q=S_q\sum_{j\neq q}^{q-1}G_j \mathrm{mod}(s_j,g_j) + s_q\), where
  \(G_j=\prod_{i}^{j-1}g_i\) (\(G_0=1\)).

One drawback of this method is that whenever the output dimensions are slightly
less than integer multiples of the input dimensions, many of the tensor elements
are repeated in an inefficient way. This is resolved be specifying that all
desired dimensions are integer multiples of the input tensor.

For example:

```prettyprint
`input` is [[ 0  1  2  3]
            [ 4  5  6  7]
            [ 8  9 10 11]]

tf.periodic_intersperse(input, [6, -1]) ==> [[ 0  1]
                                             [ 2  3]
                                             [ 4  5]
                                             [ 6  7]
                                             [ 8  9]
                                             [10 11]]
```

values: The tensor of rank `R` to periodic_intersperse
desired_shape: A 1-D tensor representing the desired shape of the output tensor.
  Exactly one element of this tensor must have the value `-1` which represents
  that this dimension of `values` can be adjusted downward in order to
  accomodate increases in other dimensions. The specified sizes of the
  non-adjustable dimensions must by at least as large as in the `values` tensor.
output: Preiodically interspersed tensor that has dimensions specified as in
  `desired_shape` except that the dimension specified as `-1` will be minimally
  decreased as necessary.

)doc");
