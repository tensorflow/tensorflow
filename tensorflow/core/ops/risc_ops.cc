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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("RiscAdd")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .SetIsAggregate()
    .SetIsCommutative();

// TODO(b/171294012): change shape function.
REGISTER_OP("RiscBroadcast")
    .Input("input: T")
    .Input("shape: Tidx")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscConcat")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ConcatV2Shape);

// TODO(b/171294012): change shape function.
REGISTER_OP("RiscConv")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::UnknownShape)
    .Attr("dilations: list(int) = [1, 1, 1, 1]");

REGISTER_OP("RiscDot")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::MatMulShape);

REGISTER_OP("RiscMax")
    .Input("x: T")
    .Input("y: T")
    .Output("max: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// TODO(b/171294012): change shape function.
REGISTER_OP("RiscPad")
    .Input("input: T")
    .Input("paddings: Tpaddings")
    .Input("constant_values: T")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tpaddings: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/171294012): change shape function.
REGISTER_OP("RiscPool")
    .Input("value: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("pooling_type: {'AVG', 'MAX'}")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/171294012): change shape function.
REGISTER_OP("RiscReshape")
    .Input("tensor: T")
    .Input("shape: Tshape")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

// TODO(b/171294012): change shape function.
REGISTER_OP("RiscShape")
    .Input("input: T")
    .Output("output: out_type")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("RiscSlice")
    .Input("input: T")
    .Input("begin: Index")
    .Input("size: Index")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Index: {int32,int64}")
    .SetShapeFn(shape_inference::SliceShape);

}  // namespace tensorflow
