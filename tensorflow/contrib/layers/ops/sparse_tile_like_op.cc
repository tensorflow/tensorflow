// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("SparseTileLike")
    .Input("a_input_indices: int64")
    .Input("a_input_values: T")
    .Input("a_input_shape: int64")
    .Input("b_input_indices: int64")
    .Input("b_input_values: T")
    .Input("b_input_shape: int64")
    .Input("axes: int32")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Tile a `SparseTensor` `a` to a new SparseTensor with the same as `SparseTensor` `b` on the axes `axes`.

The Operator takes two `SparseTensor` as inputs. `axes` means the dimensions to tile.
Here your `SparseTensor`s must have the equation `dim_a = dim_b - 1` and the shape of `b`
should have the same shape of `a` except the dimension to be tiled.

a_input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
a_input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
a_input_shape: 1-D.  Shape of the input SparseTensor.
b_input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
b_input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
b_input_shape: 1-D.  Shape of the input SparseTensor.
axes: 1-D. Length-`K` vector containing the tile axes.

For example. If `a` is a `SparseTensor` like:

```
    index = [0]
            [4]
            [6]
    values = [1, 2, 3]
    shape = [10]
```

while `b` is the template `SparseTensor` like:

```
    index = [0, 3]
            [4, 1]
            [4, 3]
            [6, 0]
            [6, 3]
    values = [8, 8, 9, 9, 9]
    shape = [10, 10]
```

with `axes=1`, then the final `SparseTensor` will be:

```
    index = [0, 3]
            [4, 1]
            [4, 3]
            [6, 0]
            [6, 3]
    values = [1, 2, 2, 3, 3]
    shape = [10, 10]
```

As you might see, the `op` will forget all the values in SparseTensor `b` and 
fill `b` with values from SparseTensor `a` at the right position.


)doc");

}  
