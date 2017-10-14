/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

// --------------------------------------------------------------------------
REGISTER_OP("Roll")
    .Input("input: T")
    .Input("shift: Tshift")
    .Input("axis: Taxis")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tshift: {int32,int64}")
    .Attr("Taxis: {int32,int64}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Rolls the elements of a tensor by the offsets of `shift` along the dimensions
of `axis`. Elements that roll passed the last position will wrap around
to the first.

For example:

```
# 't' is [0, 1, 2, 3, 4]
roll(t, shift=2, axis=0) ==> [3, 4, 0, 1, 2]

# shifting along multiple dimensions
# 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
roll(t, shift=[1, -2], axis=[0, 1]) ==> [[7, 8, 9, 5, 6], [2, 3, 4, 0, 1]]

# shifting along the same axis multiple times
# 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
roll(t, shift=[2, -3], axis=[1, 1]) ==> [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]
```

shift: `shift[i]` specifies the number of places by which elements are shifted
  along the dimension specified by `axis[i]`. Negative shifts will roll the
  elements in the opposite direction.
axis: `axis[i]` specifies the dimension that the shift `shift[i]` should occur.
  if the same axis is referenced more than once, the total shift for that axis
  will be the sum of all the shifts that belong to that axis.
output: Has the same shape and size as the input. The elements are shifted by
  the offsets of `shift` along the dimensions of `axis`.
)doc");

}  // namespace tensorflow
