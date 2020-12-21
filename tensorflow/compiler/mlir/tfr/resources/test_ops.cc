/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("TestNoOp");

REGISTER_OP("TestIdentityOp")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: numbertype");

REGISTER_OP("TestIdentityNOp")
    .Input("input: N * T")
    .Output("output: N * T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype");

REGISTER_OP("TestInputNOp")
    .Input("input: N * T")
    .Output("output: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype");

REGISTER_OP("TestOutputNOp")
    .Input("input: T")
    .Output("output: N * T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype");

REGISTER_OP("TestTwoInputsOp")
    .Input("lhs: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("pred: bool = false");

REGISTER_OP("TestComplexTFOp")
    .Input("lhs: T")
    .Input("rhs: Tlen")
    .Output("output: N * T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .Attr("Tlen: {int32, int64} = DT_INT64");

REGISTER_OP("TestNumAttrsOp")
    .Attr("x1: int = -10")
    .Attr("y1: int = 1")
    .Attr("x2: float = 0.0")
    .Attr("y2: float = -3.0");

REGISTER_OP("TestNonNumAttrsOp")
    .Attr("z: shape")
    .Attr("x: string = 'hello'")
    .Attr("y: type = DT_FLOAT");

REGISTER_OP("TestThreeInputsOp")
    .Input("x: T")
    .Input("y: T")
    .Input("z: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("act: {'x', 'y', 'z'} = 'z'");

REGISTER_OP("TestTwoOutputsOp")
    .Input("input: T")
    .Output("output1: T")
    .Output("output2: T")
    .Attr("T: numbertype");

}  // namespace tensorflow
