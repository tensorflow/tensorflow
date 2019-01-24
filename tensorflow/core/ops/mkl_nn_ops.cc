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
    .Attr("T: {float}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = false")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. MKL DNN graph transformer
 is expected to create these operators.
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
    .Attr("T: {float}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    .Attr("Tpaddings: {int32, int64} = DT_INT32")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
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
    .Attr("T: {float}")
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
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::Conv2DShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. MKL DNN graph transformer
 is expected to create these operators.
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

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
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

NOTE Do not invoke this operator directly in Python. Graph rewrite pass is
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
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
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
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
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
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
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
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
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
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
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
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
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
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
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
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
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
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
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

}  // namespace tensorflow

#endif  // INTEL_MKL
