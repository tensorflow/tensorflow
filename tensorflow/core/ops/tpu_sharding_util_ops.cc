/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("XlaSplitND")
    .Input("input: T")
    .Output("outputs: N * T")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Attr("num_splits: list(int)")
    .Attr("paddings: list(int) = []")
    // TODO(lyandy): Define shape inference function.
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("XlaConcatND")
    .Input("inputs: N * T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Attr("num_concats: list(int)")
    .Attr("paddings: list(int) = []")
    // TODO(lyandy): Define shape inference function.
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ReadVariableXlaSplitND")
    .Input("resource: resource")
    .Output("outputs: N * T")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Attr("num_splits: list(int)")
    .Attr("paddings: list(int) = []")
    // TODO(lyandy): Define shape inference function.
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("AssignVariableXlaConcatND")
    .Input("resource: resource")
    .Input("inputs: N * T")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Attr("num_concats: list(int)")
    .Attr("paddings: list(int) = []")
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow
