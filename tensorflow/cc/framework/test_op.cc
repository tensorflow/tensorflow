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

namespace tensorflow {

REGISTER_OP("ThrowAway1")
    .Input("ret: int32")
    .Input("unique_name: float")
    .Input("for: int32")
    .Attr("scope: int")
    .Attr("builder: int = 1")
    .Attr("while: int")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Op to test keywords and reserved words in input and attr names.

ret: Return value.
for: Keyword as name for input.
while: Keyword as name for attr.
)doc");

REGISTER_OP("ThrowAway2")
    .Attr("scope: int = 2")
    .Attr("throw_away2: int = 2")
    .Attr("attrs: int = 4")
    .Attr("node: int = 4")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ThrowAway3")
    .Output("node: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ThrowAway4")
    .Input("node: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ThrowAway5")
    .Output("foo: int32")
    .Attr("node: int = 4")
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
