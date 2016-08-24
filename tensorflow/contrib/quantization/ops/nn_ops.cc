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
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("QuantizedAvgPool")
    .Input("input: T")
    .Input("min_input: float")
    .Input("max_input: float")
    .Output("output: T")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int)")
    .Attr("strides: list(int)")
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
Produces the average pool of the input tensor for quantized types.

input: 4-D with shape `[batch, height, width, channels]`.
ksize: The size of the window for each dimension of the input tensor.
  The length must be 4 to match the number of dimensions of the input.
strides: The stride of the sliding window for each dimension of the input
  tensor.  The length must be 4 to match the number of dimensions of the input.
padding: The type of padding algorithm to use.
min_input: The float value that the lowest quantized input value represents.
max_input: The float value that the highest quantized input value represents.
min_output: The float value that the lowest quantized output value represents.
max_output: The float value that the highest quantized output value represents.

)doc");

REGISTER_OP("QuantizedBiasAdd")
    .Input("input: T1")
    .Input("bias: T2")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_bias: float")
    .Input("max_bias: float")
    .Output("output: out_type")
    .Output("min_out: float")
    .Output("max_out: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("out_type: quantizedtype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::BiasAddShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Adds Tensor 'bias' to Tensor 'input' for Quantized types.

Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

bias: A 1D bias Tensor with size matching the last dimension of 'input'.
min_input: The float value that the lowest quantized input value represents.
max_input: The float value that the highest quantized input value represents.
min_bias: The float value that the lowest quantized bias value represents.
max_bias: The float value that the highest quantized bias value represents.
min_out: The float value that the lowest quantized output value represents.
max_out: The float value that the highest quantized output value represents.

)doc");

REGISTER_OP("QuantizedConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
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
    })
    .Doc(R"doc(
Computes a 2D convolution given quantized 4D input and filter tensors.
The inputs are quantized tensors where the lowest value represents the real
number of the associated minimum, and the highest represents the maximum.
This means that you can only interpret the quantized output in the same way, by
taking the returned minimum and maximum values into account.

filter: filter's input_depth dimension must match input's depth dimensions.
strides: The stride of the sliding window for each dimension of the input
  tensor.
padding: The type of padding algorithm to use.
min_input: The float value that the lowest quantized input value represents.
max_input: The float value that the highest quantized input value represents.
min_filter: The float value that the lowest quantized filter value represents.
max_filter: The float value that the highest quantized filter value represents.
min_output: The float value that the lowest quantized output value represents.
max_output: The float value that the highest quantized output value represents.

)doc");

REGISTER_OP("QuantizedMaxPool")
    .Input("input: T")
    .Input("min_input: float")
    .Input("max_input: float")
    .Output("output: T")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int)")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Produces the max pool of the input tensor for quantized types.

input: The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
ksize: The size of the window for each dimension of the input tensor.
  The length must be 4 to match the number of dimensions of the input.
strides: The stride of the sliding window for each dimension of the input
  tensor. The length must be 4 to match the number of dimensions of the input.
padding: The type of padding algorithm to use.
min_input: The float value that the lowest quantized input value represents.
max_input: The float value that the highest quantized input value represents.
min_output: The float value that the lowest quantized output value represents.
max_output: The float value that the highest quantized output value represents.

)doc");

REGISTER_OP("QuantizedRelu")
    .Input("features: Tinput")
    .Input("min_features: float")
    .Input("max_features: float")
    .Output("activations: out_type")
    .Output("min_activations: float")
    .Output("max_activations: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Computes Quantized Rectified Linear: `max(features, 0)`

activations: Has the same output shape as "features".
min_features: The float value that the lowest quantized value represents.
max_features: The float value that the highest quantized value represents.
min_activations: The float value that the lowest quantized value represents.
max_activations: The float value that the highest quantized value represents.

)doc");

REGISTER_OP("QuantizedRelu6")
    .Input("features: Tinput")
    .Input("min_features: float")
    .Input("max_features: float")
    .Output("activations: out_type")
    .Output("min_activations: float")
    .Output("max_activations: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`

activations: Has the same output shape as "features".
min_features: The float value that the lowest quantized value represents.
max_features: The float value that the highest quantized value represents.
min_activations: The float value that the lowest quantized value represents.
max_activations: The float value that the highest quantized value represents.

)doc");

REGISTER_OP("QuantizedReluX")
    .Input("features: Tinput")
    .Input("max_value: float")
    .Input("min_features: float")
    .Input("max_features: float")
    .Output("activations: out_type")
    .Output("min_activations: float")
    .Output("max_activations: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`

activations: Has the same output shape as "features".
min_features: The float value that the lowest quantized value represents.
max_features: The float value that the highest quantized value represents.
min_activations: The float value that the lowest quantized value represents.
max_activations: The float value that the highest quantized value represents.

)doc");

REGISTER_OP("QuantizedBatchNormWithGlobalNormalization")
    .Input("t: Tinput")
    .Input("t_min: float")
    .Input("t_max: float")
    .Input("m: Tinput")
    .Input("m_min: float")
    .Input("m_max: float")
    .Input("v: Tinput")
    .Input("v_min: float")
    .Input("v_max: float")
    .Input("beta: Tinput")
    .Input("beta_min: float")
    .Input("beta_max: float")
    .Input("gamma: Tinput")
    .Input("gamma_min: float")
    .Input("gamma_max: float")
    .Output("result: out_type")
    .Output("result_min: float")
    .Output("result_max: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype")
    .Attr("variance_epsilon: float")
    .Attr("scale_after_normalization: bool")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

      DimensionHandle last_dim = c->Dim(input, 3);
      for (int i = 1; i < 5; ++i) {  // covers m, v, beta, gamma
        ShapeHandle vec;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i * 3), 1, &vec));
        TF_RETURN_IF_ERROR(c->Merge(last_dim, c->Dim(vec, 0), &last_dim));
      }

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 3, last_dim, &out));
      c->set_output(0, out);
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());

      return Status::OK();
    })
    .Doc(R"doc(
Quantized Batch normalization.

This op is deprecated and will be removed in the future. Prefer
`tf.nn.batch_normalization`.

t: A 4D input Tensor.
t_min: The value represented by the lowest quantized input.
t_max: The value represented by the highest quantized input.
m: A 1D mean Tensor with size matching the last dimension of t.
  This is the first output from tf.nn.moments,
  or a saved moving average thereof.
m_min: The value represented by the lowest quantized mean.
m_max: The value represented by the highest quantized mean.
v: A 1D variance Tensor with size matching the last dimension of t.
  This is the second output from tf.nn.moments,
  or a saved moving average thereof.
v_min: The value represented by the lowest quantized variance.
v_max: The value represented by the highest quantized variance.
beta: A 1D beta Tensor with size matching the last dimension of t.
  An offset to be added to the normalized tensor.
beta_min: The value represented by the lowest quantized offset.
beta_max: The value represented by the highest quantized offset.
gamma: A 1D gamma Tensor with size matching the last dimension of t.
  If "scale_after_normalization" is true, this tensor will be multiplied
  with the normalized tensor.
gamma_min: The value represented by the lowest quantized gamma.
gamma_max: The value represented by the highest quantized gamma.
variance_epsilon: A small float number to avoid dividing by 0.
scale_after_normalization: A bool indicating whether the resulted tensor
  needs to be multiplied with gamma.
)doc");

}  // namespace tensorflow
