#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Shuffle")
    .Attr("T: numbertype")
    .Attr("S: {int32, int64}")
    .Input("values: T")
    .Input("desired_shape: S")
    .Output("output: T")
    .Doc(R"doc(
Shuffles tensor to best conform to `desired shape`.

This function implements a slightly more generic version of the subpixel
shuffling found in the [original paper](https://arxiv.org/abs/1609.05158).

For example:

```prettyprint
`input` is [[ 0  1  2  3]
            [ 4  5  6  7]
            [ 8  9 10 11]]

tf.shuffle(input, [6, -1]) ==> [[ 0  1]
                                [ 2  3]
                                [ 4  5]
                                [ 6  7]
                                [ 8  9]
                                [10 11]]
```

values: The tensor of rank `R` to shuffle
desired_shape: A 1-D tensor representing the desired shape of the output tensor.
  Exactly one element of this tensor must have the value `-1` which represents
  that this dimension of `values` can be adjusted downward in order to
  accomodate increases in other dimensions. The specified sizes of the
  non-adjustable dimensions must by at least as large as in the `values` tensor.
output: Shuffled tensor that has dimensions specified as in `desired_shape`
  except that the dimension specified as `-1` will be minimally decreased as
  necessary.

)doc");
