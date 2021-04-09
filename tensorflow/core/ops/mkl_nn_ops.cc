/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

// For now, this file only includes MKL quantized ops. In the
// future, we will move all other MKL ops from nn_ops.cc to this file.

#ifdef INTEL_MKL

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("_MklNativeConv3D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("strides: list(int) >= 5")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv3DShape)
    .Doc(R"doc(
MKL version of Conv3D operator that does not depend on layout propagation.
Uses oneDNN APIs to perform 3D convolution.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeConv3DBackpropInputV2")
    .Input("input_sizes: Tshape")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("strides: list(int) >= 5")
    .Attr("dilations: list(int) = [1, 1, 1, 1, 1]")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 5, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of Convolution3D backward input op that does not depend on layout
propagation. Uses oneDNN APIs to compute the gradients of convolution with
respect to the input.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeConv3DBackpropFilterV2")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 5, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of Conv3DBackpropFilter op that does not depend on layout
propagation. Uses oneDNN APIs to compute the gradients of convolution
with respect to the filter.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeDepthwiseConv2dNative")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShapeWithExplicitPadding);

REGISTER_OP("_MklNativeDepthwiseConv2dNativeBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

REGISTER_OP("_MklNativeDepthwiseConv2dNativeBackpropFilter")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

REGISTER_OP("_MklFusedConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_args: num_args * uint8")
    .Output("output: T")
    .Output("filter_output: T")
    .Output("mkl_output: uint8")
    .Output("mkl_filter_output: uint8")
    .Attr("T: {bfloat16, float}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. MKL DNN graph transformer
 is expected to create these operators.
)doc");

REGISTER_OP("_MklNativeFusedConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. oneDNN graph transformer
 is expected to create these operators.
)doc");

REGISTER_OP("_MklNativeConv2DWithBias")
    .Input("input: T")
    .Input("filter: T")
    .Input("bias: T")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding)
    .Doc(R"doc(
MKL version of Conv2D and BiasAdd operator. Uses oneDNN APIs to perform
2D convolution and add Bias to the output of convolution.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke this operator.
)doc");

REGISTER_OP("_MklFusedDepthwiseConv2dNative")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_args: num_args * uint8")
    .Output("output: T")
    .Output("filter_output: T")
    .Output("mkl_output: uint8")
    .Output("mkl_filter_output: uint8")
    .Attr("T: {bfloat16, float}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape);

REGISTER_OP("_MklNativeFusedDepthwiseConv2dNative")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape);

REGISTER_OP("_MklFusedMatMul")
    .Input("a: T")
    .Input("b: T")
    .Input("args: num_args * T")
    .Input("mkl_a: uint8")
    .Input("mkl_b: uint8")
    .Input("mkl_args: num_args * uint8")
    .Output("product: T")
    .Output("mkl_product: uint8")
    .Attr("is_filter_const: bool = false")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, float}")
    .Attr("num_args: int >= 0")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::MatMulShape)
    .Doc(R"doc(
MKL version of FusedMatMul operator. Uses MKL-DNN APIs to implement MatMul
operator.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeFusedMatMul")
    .Input("a: T")
    .Input("b: T")
    .Input("args: num_args * T")
    .Output("product: T")
    .Attr("is_filter_const: bool = false")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, float}")
    .Attr("num_args: int >= 0")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::MatMulShape)
    .Doc(R"doc(
oneDNN version of FusedMatMul operator that does not depend
on layout propagation. Uses oneDNN APIs to implement MatMul fusion.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke this one.
)doc");

REGISTER_OP("__MklDummyPadWithFusedConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Input("paddings: Tpaddings")
    .Output("output: T")
    .Output("filter_output: T")
    .Output("mkl_output: uint8")
    .Output("mkl_filter_output: uint8")
    .Attr("T: {bfloat16, float}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    .Attr("Tpaddings: {int32, int64} = DT_INT32")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. MKL DNN graph transformer
 is expected to create these operators.
)doc");

REGISTER_OP("_MklPadWithFusedConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Input("paddings: Tpaddings")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_args: num_args * uint8")
    .Input("mkl_paddings: uint8")
    .Output("output: T")
    .Output("filter_output: T")
    .Output("mkl_output: uint8")
    .Output("mkl_filter_output: uint8")
    .Attr("T: {bfloat16, float}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    .Attr("Tpaddings: {int32, int64} = DT_INT32")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. MKL DNN graph transformer
 is expected to create these operators.
)doc");

REGISTER_OP("_MklNativePadWithFusedConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Input("paddings: Tpaddings")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    .Attr("Tpaddings: {int32, int64} = DT_INT32")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. oneDNN graph transformer
 is expected to create these operators.
)doc");

REGISTER_OP("_MklNativePadWithConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("paddings: Tpaddings")
    .Output("output: T")
    .Attr("T: {bfloat16, float}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("is_filter_const: bool = false")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("Tpaddings: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
MKL version of Pad and Conv2D fusion that does not depend
on layout propagation. Uses oneDNN APIs to perform
the fusion.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeAvgPool")
    .Input("value: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {float, half, double, bfloat16}")
    .SetShapeFn(shape_inference::AvgPoolShape)
    .Doc(R"doc(
oneDNN version of AvgPool operator that does not depend on layout
propagation. Uses oneDNN APIs to perform average pooling on the input.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeAvgPoolGrad")
    .Input("orig_input_shape: int32")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("T: {float, half, double, bfloat16}")
    .SetShapeFn(shape_inference::AvgPoolGradShape)
    .Doc(R"doc(
oneDNN version of AvgPoolGrad operator that does not depend on layout
propagation. Uses oneDNN APIs to compute gradients of AvgPool operator.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeAvgPool3D")
    .Input("value: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("T: {float, half, double, bfloat16}")
    .SetShapeFn(shape_inference::Pool3DShape)
    .Doc(R"doc(
oneDNN version of AvgPool3D operator that does not depend on layout
propagation. Uses oneDNN APIs to perform 3D average pooling on the input.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeAvgPool3DGrad")
    .Input("orig_input_shape: int32")
    .Input("grad: T")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("T: {float, half, double, bfloat16}")
    .SetShapeFn(shape_inference::AvgPool3DGradShape)
    .Doc(R"doc(
oneDNN version of AvgPool3DGrad operator that does not depend on layout
propagation. Uses oneDNN APIs to compute gradients of AvgPool3D function.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeMaxPool")
    .Attr("T: {float, half, bfloat16} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("workspace_enabled: bool = false")
    .Input("input: T")
    .Output("output: T")
    .Output("workspace: uint8")
    .SetShapeFn(shape_inference::MaxPoolShape)
    .Doc(R"doc(
oneDNN version of MaxPool operator that does not depend
on layout propagation. Uses oneDNN APIs to perform max pooling
on the input.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeMaxPoolGrad")
    .Attr("T: {float, half, bfloat16} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("workspace_enabled: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Input("orig_input: T")
    .Input("orig_output: T")
    .Input("grad: T")
    .Input("workspace: uint8")
    .Output("output: T")
    .SetShapeFn(shape_inference::MaxPoolGradShape)
    .Doc(R"doc(
oneDNN version of MaxPoolGrad that does not depend on layout propagation.
Uses oneDNN APIs to compute gradients of MaxPool operator.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeMaxPool3D")
    .Input("input: T")
    .Output("output: T")
    .Output("workspace: uint8")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("T: {half, bfloat16, float}")
    .Attr("workspace_enabled: bool = false")
    .SetShapeFn(shape_inference::Pool3DShape)
    .Doc(R"doc(
oneDNN version of MaxPool3D operator that does not depend on layout propagation.
Uses oneDNN APIs to perform 3D max pooling on the input.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeMaxPool3DGrad")
    .Input("orig_input: TInput")
    .Input("orig_output: TInput")
    .Input("grad: T")
    .Input("workspace: uint8")
    .Output("output: T")
    .Attr("ksize: list(int) >= 5")
    .Attr("strides: list(int) >= 5")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnet3dDataFormatAttrString())
    .Attr("T: {half, bfloat16, float} = DT_FLOAT")
    .Attr("TInput: {half, bfloat16, float} = DT_FLOAT")
    .Attr("workspace_enabled: bool = false")
    .SetShapeFn(shape_inference::MaxPool3DGradShape)
    .Doc(R"doc(
oneDNN version of MaxPool3DGrad operator that does not depend on layout
propagation. Uses oneDNN APIs to compute gradients of MaxPool3D function.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklQuantizedMaxPool")
    .Input("input:         T")
    .Input("min_input:     float")
    .Input("max_input:     float")
    .Input("mkl_input:     uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Output("output:       T")
    .Output("min_output:   float")
    .Output("max_output:   float")
    .Output("mkl_output:     uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .SetShapeFn(shape_inference::MaxPoolShape)
    .Doc(R"doc(
MKL version of QuantizedMaxPool operator. Uses MKL DNN APIs to perform max pooling
on the quantized input.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklQuantizedAvgPool")
    .Input("input:           T")
    .Input("min_input:       float")
    .Input("max_input:       float")
    .Input("mkl_input:       uint8")
    .Input("mkl_min_input:   uint8")
    .Input("mkl_max_input:   uint8")
    .Output("output:         T")
    .Output("min_output:     float")
    .Output("max_output:     float")
    .Output("mkl_output:     uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::AvgPoolShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
MKL version of QuantizedAvgPool operator. Uses MKL DNN APIs to perform average pooling
on the quantized input.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklQuantizedConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for enabling MklToTf
                               // conversion
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

// TODO(nammbash): Most of the  TF_RETURN_IF_ERROR(c->WithRank) checks
// seems to be similar and hence can be moved into a single function
// with appropriate arguments for a cleaner design.
REGISTER_OP("_MklQuantizedConv2DAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Input("mkl_min_freezed_output: uint8")
    .Input("mkl_max_freezed_output: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for enabling MklToTf
                               // conversion
    .Attr("out_type: quantizedtype = DT_QINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DWithBias")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: float")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling MklToTf conversion
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused, channel;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &channel));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &channel));
      c->set_output(1, channel);
      c->set_output(2, channel);
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DWithBiasAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Input("mkl_min_freezed_output: uint8")
    .Input("mkl_max_freezed_output: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling MklToTf conversion
    .Attr("out_type: quantizedtype = DT_QINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DAndRelu")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for enabling MklToTf
                               // conversion
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused, channel;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &channel));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &channel));
      c->set_output(1, channel);
      c->set_output(2, channel);
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Input("mkl_min_freezed_output: uint8")
    .Input("mkl_max_freezed_output: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for enabling MklToTf
                               // conversion
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DWithBiasAndRelu")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: float")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling MklToTf conversion
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused, channel;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &channel));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &channel));
      c->set_output(1, channel);
      c->set_output(2, channel);
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DWithBiasAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Input("mkl_min_freezed_output: uint8")
    .Input("mkl_max_freezed_output: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling MklToTf conversion
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DWithBiasSumAndRelu")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: float")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("summand: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Input("mkl_summand: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling MklToTf conversion
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused, channel;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &channel));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &channel));
      c->set_output(1, channel);
      c->set_output(2, channel);
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DWithBiasSumAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("summand: Tsummand")
    .Input("min_summand: float")
    .Input("max_summand: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Input("mkl_min_freezed_output: uint8")
    .Input("mkl_max_freezed_output: uint8")
    .Input("mkl_summand: uint8")
    .Input("mkl_min_summand: uint8")
    .Input("mkl_max_summand: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("Tsummand: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling MklToTf conversion
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("summand: Tsummand")
    .Input("min_summand: float")
    .Input("max_summand: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Input("mkl_min_freezed_output: uint8")
    .Input("mkl_max_freezed_output: uint8")
    .Input("mkl_summand: uint8")
    .Input("mkl_min_summand: uint8")
    .Input("mkl_max_summand: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("Tsummand: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling MklToTf conversion
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedConv2DPerChannel")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for enabling MklToTf
                               // conversion
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused, channel;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &channel));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &channel));
      c->set_output(1, channel);
      c->set_output(2, channel);
      return Status::OK();
    })
    .Doc(R"doc(
MKL-DNN implementation of QuantizedConv2D op.
)doc");

REGISTER_OP("_MklDepthwiseConv2dNativeBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_out_backprop: uint8")
    .Output("output: T")
    .Output("mkl_output: uint8")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

REGISTER_OP("_MklDepthwiseConv2dNativeBackpropFilter")
    .Input("input: T")
    .Input("filter_sizes: int32")
    .Input("out_backprop: T")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_out_backprop: uint8")
    .Output("output: T")
    .Output("mkl_output: uint8")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedMatMulWithBias")
    .Input("a: T1")
    .Input("b: T2")
    .Input("bias: Tbias")
    .Input("min_a: float")
    .Input("max_a: float")
    .Input("min_b: float")
    .Input("max_b: float")
    .Input("mkl_a: uint8")      // MKL second tensor
    .Input("mkl_b: uint8")      // MKL second tensor
    .Input("mkl_bias: uint8")   // MKL second tensor
    .Input("mkl_min_a: uint8")  // MKL second tensor
    .Input("mkl_max_a: uint8")  // MKL second tensor
    .Input("mkl_min_b: uint8")  // MKL second tensor
    .Input("mkl_max_b: uint8")  // MKL second tensor
    .Output("out: Toutput")
    .Output("min_out: float")
    .Output("max_out: float")
    .Output("mkl_out: uint8")      // MKL second tensor
    .Output("mkl_min_out: uint8")  // MKL second tensor
    .Output("mkl_max_out: uint8")  // MKL second tensor
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("T: quantizedtype")  // Additional attr "T" for MklToTf conversion
    .Attr("Toutput: quantizedtype = DT_QINT32")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'")
    .Attr("is_weight_const: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MatMulShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));

      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedMatMulWithBiasAndRelu")
    .Input("a: T1")
    .Input("b: T2")
    // TODO(intel-tf): Modify bias type as Tbias and add relevant attribute.
    .Input("bias: float")
    .Input("min_a: float")
    .Input("max_a: float")
    .Input("min_b: float")
    .Input("max_b: float")
    .Input("mkl_a: uint8")      // MKL second tensor
    .Input("mkl_b: uint8")      // MKL second tensor
    .Input("mkl_bias: uint8")   // MKL second tensor
    .Input("mkl_min_a: uint8")  // MKL second tensor
    .Input("mkl_max_a: uint8")  // MKL second tensor
    .Input("mkl_min_b: uint8")  // MKL second tensor
    .Input("mkl_max_b: uint8")  // MKL second tensor
    .Output("out: Toutput")
    .Output("min_out: float")
    .Output("max_out: float")
    .Output("mkl_out: uint8")      // MKL second tensor
    .Output("mkl_min_out: uint8")  // MKL second tensor
    .Output("mkl_max_out: uint8")  // MKL second tensor
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attr "T" for MklToTf conversion
    .Attr("Toutput: quantizedtype = DT_QINT32")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'")
    .Attr("is_weight_const: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MatMulShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));

      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedMatMulWithBiasAndReluAndRequantize")
    .Input("a: T1")
    .Input("b: T2")
    .Input("bias: Tbias")
    .Input("min_a: float")
    .Input("max_a: float")
    .Input("min_b: float")
    .Input("max_b: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("mkl_a: uint8")                   // MKL second tensor
    .Input("mkl_b: uint8")                   // MKL second tensor
    .Input("mkl_bias: uint8")                // MKL second tensor
    .Input("mkl_min_a: uint8")               // MKL second tensor
    .Input("mkl_max_a: uint8")               // MKL second tensor
    .Input("mkl_min_b: uint8")               // MKL second tensor
    .Input("mkl_max_b: uint8")               // MKL second tensor
    .Input("mkl_min_freezed_output: uint8")  // MKL second tensor
    .Input("mkl_max_freezed_output: uint8")  // MKL second tensor
    .Output("out: Toutput")
    .Output("min_out: float")
    .Output("max_out: float")
    .Output("mkl_out: uint8")      // MKL second tensor
    .Output("mkl_min_out: uint8")  // MKL second tensor
    .Output("mkl_max_out: uint8")  // MKL second tensor
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("T: quantizedtype")  // Additional attr "T" for MklToTf conversion
    .Attr("Toutput: quantizedtype = DT_QUINT8")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'")
    .Attr("is_weight_const: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MatMulShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));

      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedMatMulWithBiasAndDequantize")
    .Input("a: T1")
    .Input("b: T2")
    .Input("bias: Tbias")
    .Input("min_a: float")
    .Input("max_a: float")
    .Input("min_b: float")
    .Input("max_b: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("mkl_a: uint8")                   // MKL second tensor
    .Input("mkl_b: uint8")                   // MKL second tensor
    .Input("mkl_bias: uint8")                // MKL second tensor
    .Input("mkl_min_a: uint8")               // MKL second tensor
    .Input("mkl_max_a: uint8")               // MKL second tensor
    .Input("mkl_min_b: uint8")               // MKL second tensor
    .Input("mkl_max_b: uint8")               // MKL second tensor
    .Input("mkl_min_freezed_output: uint8")  // MKL second tensor
    .Input("mkl_max_freezed_output: uint8")  // MKL second tensor
    .Output("out: Toutput")
    .Output("mkl_out: uint8")  // MKL second tensor
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("T: quantizedtype")  // Additional attr "T" for MklToTf conversion
    .Attr("Toutput: {float}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'")
    .Attr("is_weight_const: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MatMulShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));

      return Status::OK();
    });

REGISTER_OP("_MklQuantizedMatMulWithBiasAndRequantize")
    .Input("a: T1")
    .Input("b: T2")
    .Input("bias: Tbias")
    .Input("min_a: float")
    .Input("max_a: float")
    .Input("min_b: float")
    .Input("max_b: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("mkl_a: uint8")                   // MKL second tensor
    .Input("mkl_b: uint8")                   // MKL second tensor
    .Input("mkl_bias: uint8")                // MKL second tensor
    .Input("mkl_min_a: uint8")               // MKL second tensor
    .Input("mkl_max_a: uint8")               // MKL second tensor
    .Input("mkl_min_b: uint8")               // MKL second tensor
    .Input("mkl_max_b: uint8")               // MKL second tensor
    .Input("mkl_min_freezed_output: uint8")  // MKL second tensor
    .Input("mkl_max_freezed_output: uint8")  // MKL second tensor
    .Output("out: Toutput")
    .Output("min_out: float")
    .Output("max_out: float")
    .Output("mkl_out: uint8")      // MKL second tensor
    .Output("mkl_min_out: uint8")  // MKL second tensor
    .Output("mkl_max_out: uint8")  // MKL second tensor
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("T: quantizedtype")  // Additional attr "T" for MklToTf conversion
    .Attr("Toutput: quantizedtype = DT_QUINT8")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("input_quant_mode: {'MIN_FIRST', 'SCALED'} = 'MIN_FIRST'")
    .Attr("is_weight_const: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MatMulShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));

      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("_MklQuantizedDepthwiseConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    // In order to enable MKL to TF conversion, _MklToTf op requires the
    // attribute "T" to be specified.
    .Attr("T: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      // TODO(bhavanis): Print an error message during the return.
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused, channel;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &channel));
      c->set_output(1, channel);
      c->set_output(2, channel);
      return Status::OK();
    })
    .Doc(R"doc(
MKL-DNN implementation of quantized depthwise Conv2D.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke this operator.
)doc");

REGISTER_OP("_MklQuantizedDepthwiseConv2DWithBias")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: float")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    // Additional attribute "T" for enabling MKL to TF conversion
    .Attr("T: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused, channel;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &channel));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &channel));
      c->set_output(1, channel);
      c->set_output(2, channel);
      return Status::OK();
    })
    .Doc(R"doc(
MKL-DNN implementation of quantized depthwise Conv2D with Bias.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke this operator.
)doc");

REGISTER_OP("_MklQuantizedDepthwiseConv2DWithBiasAndRelu")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: float")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    // Additional attribute "T" for enabling MKL to TF conversion
    .Attr("T: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused, channel;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &channel));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &channel));
      c->set_output(1, channel);
      c->set_output(2, channel);
      return Status::OK();
    })
    .Doc(R"doc(
MKL-DNN implementation of quantized depthwise Conv2D with Bias and Relu.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke this operator.
)doc");

REGISTER_OP("_MklQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("mkl_input: uint8")
    .Input("mkl_filter: uint8")
    .Input("mkl_bias: uint8")
    .Input("mkl_min_input: uint8")
    .Input("mkl_max_input: uint8")
    .Input("mkl_min_filter: uint8")
    .Input("mkl_max_filter: uint8")
    .Input("mkl_min_freezed_output: uint8")
    .Input("mkl_max_freezed_output: uint8")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Output("mkl_output: uint8")
    .Output("mkl_min_output: uint8")
    .Output("mkl_max_output: uint8")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    // Additional attribute "T" for enabling MKL to TF conversion
    .Attr("T: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr("is_bias_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
MKL-DNN implementation of quantized depthwise Conv2D with Bias, Relu and Requantize.
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke this operator.
)doc");

REGISTER_OP("_MklFusedBatchNormV3")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Input("mean: U")
    .Input("variance: U")
    .Input("mkl_x: uint8")
    .Input("mkl_scale: uint8")
    .Input("mkl_offset: uint8")
    .Input("mkl_mean: uint8")
    .Input("mkl_variance: uint8")
    .Output("y: T")
    .Output("batch_mean: U")
    .Output("batch_variance: U")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Output("reserve_space_3: U")
    .Output("mkl_y: uint8")
    .Output("mkl_batch_mean: uint8")
    .Output("mkl_batch_variance: uint8")
    .Output("mkl_reserve_space_1: uint8")
    .Output("mkl_reserve_space_2: uint8")
    .Output("mkl_reserve_space_3: uint8")
    .Attr("T: {half, bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("exponential_avg_factor: float = 1.0")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormShape)
    .Doc(
        R"doc(MKL-DNN implementation of FusedBatchNormV3: Do not invoke this operator directly in Python.
         Graph rewrite pass is expected to invoke this operator.)doc");

REGISTER_OP("_MklFusedBatchNormGradV3")
    .Input("y_backprop: T")
    .Input("x: T")
    .Input("scale: float")
    .Input("reserve_space_1: U")
    .Input("reserve_space_2: U")
    .Input("reserve_space_3: U")
    .Input("mkl_y_backprop: uint8")
    .Input("mkl_x: uint8")
    .Input("mkl_scale: uint8")
    .Input("mkl_reserve_space_1: uint8")
    .Input("mkl_reserve_space_2: uint8")
    .Input("mkl_reserve_space_3: uint8")
    .Output("x_backprop: T")
    .Output("scale_backprop: U")
    .Output("offset_backprop: U")
    .Output("reserve_space_4: U")
    .Output("reserve_space_5: U")
    .Output("mkl_x_backprop: uint8")
    .Output("mkl_scale_backprop: uint8")
    .Output("mkl_offset_backprop: uint8")
    .Output("mkl_reserve_space_4: uint8")
    .Output("mkl_reserve_space_5: uint8")
    .Attr("T: {half, bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormGradShape)
    .Doc(
        R"doc(MKL-DNN implementation of FusedBatchNormGradV3: Do not invoke this operator directly in Python.
             Graph rewrite pass is expected to invoke this operator.)doc");

REGISTER_OP("_MklFusedBatchNormEx")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Input("mean: U")
    .Input("variance: U")
    .Input("side_input: num_side_inputs * T")
    .Input("mkl_x: uint8")
    .Input("mkl_scale: uint8")
    .Input("mkl_offset: uint8")
    .Input("mkl_mean: uint8")
    .Input("mkl_variance: uint8")
    .Input("mkl_side_input: num_side_inputs * uint8")
    .Output("y: T")
    .Output("batch_mean: U")
    .Output("batch_variance: U")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Output("reserve_space_3: U")
    .Output("mkl_y: uint8")
    .Output("mkl_batch_mean: uint8")
    .Output("mkl_batch_variance: uint8")
    .Output("mkl_reserve_space_1: uint8")
    .Output("mkl_reserve_space_2: uint8")
    .Output("mkl_reserve_space_3: uint8")
    .Attr("T: {bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr("exponential_avg_factor: float = 1.0")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("num_side_inputs: int >= 0 = 0")
    .Attr("activation_mode: string = \"Identity\"")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormShape)
    .Doc(R"doc(
MKL version of FusedBatchNormEx operator. Uses MKL DNN APIs to perform fused
batch normalization and relu.

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeFusedBatchNorm")
    .Input("x: T")
    .Input("scale: T")
    .Input("offset: T")
    .Input("mean: T")
    .Input("variance: T")
    .Output("y: T")
    .Output("batch_mean: T")
    .Output("batch_variance: T")
    .Output("reserve_space_1: T")
    .Output("reserve_space_2: T")
    .Attr("T: numbertype")
    .Attr("epsilon: float = 0.0001")
    .Attr("data_format: string = 'NHWC'")
    .Attr("exponential_avg_factor: float = 1.0")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormShape)
    .Doc(R"doc(
oneDNN version of FusedBatchNorm operator that does not depend on layout
propagation. Uses oneDNN APIs to perform fused batch normalization.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeFusedBatchNormGrad")
    .Input("y_backprop: T")
    .Input("x: T")
    .Input("scale: T")
    .Input("reserve_space_1: T")
    .Input("reserve_space_2: T")
    .Output("x_backprop: T")
    .Output("scale_backprop: T")
    .Output("offset_backprop: T")
    .Output("reserve_space_3: T")
    .Output("reserve_space_4: T")
    .Attr("T: numbertype")
    .Attr("epsilon: float = 0.0001")
    .Attr("data_format: string = 'NHWC'")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormGradShape)
    .Doc(R"doc(
oneDNN version of FusedBatchNormGrad operator that does not depend
on layout propagation. Uses oneDNN APIs to compute gradients for fused
batch normalization.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

REGISTER_OP("_MklNativeFusedBatchNormV2")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Input("mean: U")
    .Input("variance: U")
    .Output("y: T")
    .Output("batch_mean: U")
    .Output("batch_variance: U")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Attr("T: {bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("exponential_avg_factor: float = 1.0")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormShape);

REGISTER_OP("_MklNativeFusedBatchNormGradV2")
    .Input("y_backprop: T")
    .Input("x: T")
    .Input("scale: float")
    .Input("reserve_space_1: U")
    .Input("reserve_space_2: U")
    .Output("x_backprop: T")
    .Output("scale_backprop: U")
    .Output("offset_backprop: U")
    .Output("reserve_space_3: U")
    .Output("reserve_space_4: U")
    .Attr("T: {bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormGradShape);

REGISTER_OP("_MklNativeFusedBatchNormV3")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Input("mean: U")
    .Input("variance: U")
    .Output("y: T")
    .Output("batch_mean: U")
    .Output("batch_variance: U")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Output("reserve_space_3: U")
    .Attr("T: {half, bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("exponential_avg_factor: float = 1.0")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormShape)
    .Doc(
        R"doc(oneDNN version of FusedBatchNormV3 operator that does not depend
        on layout propagation. Do not invoke this operator directly in Python.
        Graph rewrite pass is expected to invoke this operator.)doc");

REGISTER_OP("_MklNativeFusedBatchNormGradV3")
    .Input("y_backprop: T")
    .Input("x: T")
    .Input("scale: float")
    .Input("reserve_space_1: U")
    .Input("reserve_space_2: U")
    .Input("reserve_space_3: U")
    .Output("x_backprop: T")
    .Output("scale_backprop: U")
    .Output("offset_backprop: U")
    .Output("reserve_space_4: U")
    .Output("reserve_space_5: U")
    .Attr("T: {half, bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormGradShape)
    .Doc(
        R"doc(oneDNN version of FusedBatchNormGradV3 that does not depend
        on layout propagation. Do not invoke this operator directly in Python.
        Graph rewrite pass is expected to invoke this operator.)doc");

REGISTER_OP("_MklNativeFusedBatchNormEx")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Input("mean: U")
    .Input("variance: U")
    .Input("side_input: num_side_inputs * T")
    .Output("y: T")
    .Output("batch_mean: U")
    .Output("batch_variance: U")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Output("reserve_space_3: U")
    .Attr("T: {bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr("exponential_avg_factor: float = 1.0")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("num_side_inputs: int >= 0 = 0")
    .Attr("activation_mode: string = \"Identity\"")
    .Attr("is_training: bool = true")
    .SetShapeFn(shape_inference::FusedBatchNormShape)
    .Doc(R"doc(
oneDNN version of FusedBatchNormEx operator that does not depend on layout propagation.
Uses oneDNN APIs to perform fused batch normalization and relu.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

}  // namespace tensorflow

#endif  // INTEL_MKL
