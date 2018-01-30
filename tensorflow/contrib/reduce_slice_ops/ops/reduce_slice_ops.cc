/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status ReduceSliceShapeFn(InferenceContext* c) {
  ShapeHandle handle;
  DimensionHandle dimhandle;
  DimensionHandle dim_axis = c->UnknownDim();
  // "axis" must be a scala
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &handle));
  // "data" must have rank at least 1
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &handle));
  // "indices" must have have rank 1 or rank 2 with the number of columns must
  // be 2
  if (c->RankKnown(c->input(1))) {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 2, &handle));
    if (c->Rank(c->input(1)) == 1) {
      // if "indices" is a vector of 0 elements, then the axis dimension of
      // output tensor should be of dimension 0.
      DimensionHandle raw_dim_axis;
      TF_RETURN_IF_ERROR(c->Max(c->Dim(c->input(1), 0), 1, &raw_dim_axis));
      TF_RETURN_IF_ERROR(c->Subtract(raw_dim_axis, 1, &dim_axis));
    } else {  // c->Rank(c->input(1)) == 2
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(c->input(1), 1), c->MakeDim(2), &dimhandle));
      dim_axis = c->Dim(c->input(1), 0);
    }
  }
  // shape of output tensor
  const Tensor* _axis = c->input_tensor(2);
  if (nullptr == _axis) {
    c->set_output(0, c->UnknownShapeOfRank(c->Rank(c->input(0))));
  } else {
    int64 axis = _axis->scalar<int64>()();
    TF_RETURN_IF_ERROR(c->ReplaceDim(handle, axis, dim_axis, &handle));
    c->set_output(0, handle);
  }
  return Status::OK();
}

}  // namespace

REGISTER_OP("ReduceSliceSum")
    .Input("data: T")
    .Input("indices: Tindices")
    .Input("axis: int64")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(ReduceSliceShapeFn)
    .Doc(R"doc(
Dynamically sum over the first dimension of a tensor according to start and end
indices specified at 'index'.

For example:

```prettyprint
# if 'data' is [[   1,   2,   3]
                [  40,  50,  60]
                [ 700, 800, 900]
                [1000,2000,3000]],

and 'indices' is [[0,1]
                  [1,1]
                  [0,2]],

the output will be [[ 1, 2, 3]
                    [ 0, 0, 0]
                    [41,52,63]].
```

The data must be at least rank 1. The indices must be of shape (?,2) where the
first column is start indices and the second column is end indices. The end indices
are not included in the reduce operation, which means, if you want to do a reduce
over indices 0,1,2, then you should have start index 0 and end index 3. If end
index is smaller than or equal to start, the result will be zero. If end index is
out of bounds, then the reduce operation will automatically stop at the bound, so
feel free to put a large number as your end of your index if you want to do the
reduction until the bound.

data: The source of data where the computation will be taken from.
indices: start, end indices that controls which part to be included.
T: the type of data.
Tindices: the type of indices, must be int32 or int64.
output: the computed sum values.
)doc");

REGISTER_OP("ReduceSliceProd")
    .Input("data: T")
    .Input("indices: Tindices")
    .Input("axis: int64")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(ReduceSliceShapeFn)
    .Doc(R"doc(
Dynamically compute the product over the first dimension of a tensor according
to start and end indices specified at 'indices'.

For example:

```prettyprint
# if 'data' is [[   1,   2,   3]
                [  40,  50,  60]
                [ 700, 800, 900]
                [1000,2000,3000]],

and 'indices' is [[0,1]
                  [1,1]
                  [0,2]],

the output will be [[ 1,  2,  3]
                    [ 1,  1,  1]
                    [40,100,180]].
```

The data must be at least rank 1. The indices can be of shape (?,2) where the
first column is start indices and the second column is end indices. The end indices
are not included in the reduce operation, which means, if you want to do a reduce
over indices 0,1,2, then you should have start index 0 and end index 3. If end
index is smaller than or equal to start, the result will be 1. If end index is
out of bounds, then the reduce operation will automatically stop at the bound, so
feel free to put a large number as your end of your index if you want to do the
reduction until the bound. The indices can also be of shape (?), in this case, the
start index of i will be the element at i, then end index of i will be the element
at i+1. That is:

```prettyprint
indices = [0,5,11,115]

is equivalent to

indices = [ [0,5],
            [5,11],
            [11,115]]
```

data: The source of data where the computation will be taken from.
indices: start, end indices that controls which part to be included.
T: the type of data.
Tindices: the type of indices, must be int32 or int64.
output: the computed product values.
)doc");

REGISTER_OP("ReduceSliceMax")
    .Input("data: T")
    .Input("indices: Tindices")
    .Input("axis: int64")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(ReduceSliceShapeFn)
    .Doc(R"doc(
Dynamically compute the maximum over the first dimension of a tensor according
to start and end indices specified at "indices".

For example:

```prettyprint
# if 'data' is [[   1,  20,   3]
                [ 400,   5,  60]
                [  70,   8, 900]
                [1000,2000,3000]],

and 'indices' is [[0,1]
                  [1,1]
                  [0,2]],

the output will be [[          1,         20,          3]
                    [ -BIG_VALUE, -BIG_VALUE, -BIG_VALUE]
                    [        400,         20,         60]].
```

The data must be at least rank 1. The indices can be of shape (?,2) where the
first column is start indices and the second column is end indices. The end indices
are not included in the reduce operation, which means, if you want to do a reduce
over indices 0,1,2, then you should have start index 0 and end index 3. If end
index is smaller than or equal to start, the result will be 1. If end index is
out of bounds, then the reduce operation will automatically stop at the bound, so
feel free to put a large number as your end of your index if you want to do the
reduction until the bound. The indices can also be of shape (?), in this case, the
start index of i will be the element at i, then end index of i will be the element
at i+1. That is:

```prettyprint
indices = [0,5,11,115]

is equivalent to

indices = [ [0,5],
            [5,11],
            [11,115]]
```

data: The source of data where the computation will be taken from.
indices: start, end indices that controls which part to be included.
T: the type of data.
Tindices: the type of indices, must be int32 or int64.
output: the computed product values.
)doc");

REGISTER_OP("ReduceSliceMin")
    .Input("data: T")
    .Input("indices: Tindices")
    .Input("axis: int64")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(ReduceSliceShapeFn)
    .Doc(R"doc(
Dynamically compute the minimum over the first dimension of a tensor according
to start and end indices specified at 'indices'.

For example:

```prettyprint
# if 'data' is [[   1,  20,   3]
                [ 400,   5,  60]
                [  70,   8, 900]
                [1000,2000,3000]],

and 'indices' is [[0,1]
                  [1,1]
                  [0,2]],

the the output will be [[          1,         20,          3]
                        [ +BIG_VALUE, +BIG_VALUE, +BIG_VALUE]
                        [          1,          5,          3]].
```

The data must be at least rank 1. The indices can be of shape (?,2) where the
first column is start indices and the second column is end indices. The end indices
are not included in the reduce operation, which means, if you want to do a reduce
over indices 0,1,2, then you should have start index 0 and end index 3. If end
index is smaller than or equal to start, the result will be 1. If end index is
out of bounds, then the reduce operation will automatically stop at the bound, so
feel free to put a large number as your end of your index if you want to do the
reduction until the bound. The indices can also be of shape (?), in this case, the
start index of i will be the element at i, then end index of i will be the element
at i+1. That is:

```prettyprint
indices = [0,5,11,115]

is equivalent to

indices = [ [0,5],
            [5,11],
            [11,115]]
```

data: The source of data where the computation will be taken from.
indices: start, end indices that controls which part to be included.
T: the type of data.
Tindices: the type of indices, must be int32 or int64.
output: the computed product values.
)doc");

}  // namespace tensorflow
