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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {
Status GetMeanAndVarianceSize(shape_inference::InferenceContext* c,
                              int64& num_groups_time_batches) {
  auto in_shape = c->input(0);
  // Get the number of batches.
  std::string data_format_str;
  TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format_str));
  TensorFormat data_format;
  if (!FormatFromString(data_format_str, &data_format)) {
    return tensorflow::errors::Unknown("Unknown format %s.",
                                       data_format_str.c_str());
  }
  const int batch_index =
      GetTensorBatchDimIndex(c->Rank(in_shape), data_format);
  auto batch_size = c->Value(c->Dim(in_shape, batch_index));

  int32 num_groups;
  TF_RETURN_IF_ERROR(c->GetAttr("num_groups", &num_groups));
  num_groups_time_batches = num_groups * batch_size;
  return Status::OK();
}
}  // namespace

REGISTER_OP("PopnnLstmLayer")
    .Input("inputs: dtype")
    .Input("input_h_state: dtype")
    .Input("input_c_state: dtype")
    .Input("kernel: dtype")
    .Input("biases: dtype")
    .Output("output: dtype")
    .Output("output_h_state: dtype")
    .Output("output_c_state: dtype")
    .Output("intermediates: dtype")
    .Attr("num_channels: int")
    .Attr("is_training: bool")
    .Attr("dtype: {float16, float32}")
    .Attr("partials_dtype: {float16, float32} = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 num_channels;
      TF_RETURN_IF_ERROR(c->GetAttr("num_channels", &num_channels));
      shape_inference::DimensionOrConstant doc_num_channels(num_channels);

      auto inputs = c->input(0);
      auto time_steps = c->Dim(inputs, 0);
      auto batch_size = c->Dim(inputs, 1);

      c->set_output(0,
                    c->MakeShape({time_steps, batch_size, doc_num_channels}));
      c->set_output(1, c->MakeShape({batch_size, doc_num_channels}));
      c->set_output(2, c->MakeShape({batch_size, doc_num_channels}));
      c->set_output(3, c->MakeShape({}));
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnLstmLayer.
)doc");

REGISTER_OP("PopnnLstmLayerBackprop")
    .Input("inputs: dtype")
    .Input("input_h_state: dtype")
    .Input("input_c_state: dtype")
    .Input("kernel: dtype")
    .Input("biases: dtype")
    .Input("output: dtype")
    .Input("output_h_state: dtype")
    .Input("output_c_state: dtype")
    .Input("intermediates: dtype")
    .Input("output_backprop: dtype")
    .Input("output_h_state_backprop: dtype")
    .Input("output_c_state_backprop: dtype")
    .Output("inputs_backprop: dtype")
    .Output("input_h_state_backprop: dtype")
    .Output("input_c_state_backprop: dtype")
    .Output("kernel_backprop: dtype")
    .Output("biases_backprop: dtype")
    .Attr("num_channels: int")
    .Attr("is_training: bool")
    .Attr("dtype: {float16, float32}")
    .Attr("partials_dtype: {float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      auto in_h_shape = c->input(1);
      auto in_c_shape = c->input(2);
      auto kernel_shape = c->input(3);
      auto biases_shape = c->input(4);
      c->set_output(0, in_shape);
      c->set_output(1, in_h_shape);
      c->set_output(2, in_c_shape);
      c->set_output(3, kernel_shape);
      c->set_output(4, biases_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnLstmLayerBackprop.
)doc");

REGISTER_OP("PopnnGroupNormInference")
    .Input("inputs: dtype")
    .Input("gamma: dtype")
    .Input("beta: dtype")
    .Input("mean: dtype")
    .Input("inv_std_dev: dtype")
    .Output("output: dtype")
    .Attr("data_format: string")
    .Attr("epsilon: float")
    .Attr("num_groups: int")
    .Attr("dtype: {float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      c->set_output(0, in_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnGroupNormInference.
)doc");

REGISTER_OP("PopnnGroupNormTraining")
    .Input("inputs: dtype")
    .Input("gamma: dtype")
    .Input("beta: dtype")
    .Output("output: dtype")
    .Output("mean: dtype")
    .Output("inv_std_dev: dtype")
    .Output("input_whitened: dtype")
    .Attr("data_format: string")
    .Attr("epsilon: float")
    .Attr("num_groups: int")
    .Attr("dtype: {float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);

      int64 num_groups_time_batch;
      TF_RETURN_IF_ERROR(GetMeanAndVarianceSize(c, num_groups_time_batch));
      shape_inference::DimensionOrConstant doc_num_groups_time_batch(
          num_groups_time_batch);
      auto mean_inv_std_dev_shape = c->MakeShape({doc_num_groups_time_batch});

      c->set_output(0, in_shape);
      c->set_output(1, mean_inv_std_dev_shape);
      c->set_output(2, mean_inv_std_dev_shape);
      c->set_output(3, in_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnGroupNormTraining.
)doc");

REGISTER_OP("PopnnGroupNormGrad")
    .Input("input_whitened: dtype")
    .Input("gamma: dtype")
    .Input("mean: dtype")
    .Input("inv_std_dev: dtype")
    .Input("output_backprop: dtype")
    .Output("inputs_backprop: dtype")
    .Output("gamma_backprop: dtype")
    .Output("beta_backprop: dtype")
    .Attr("data_format: string")
    .Attr("epsilon: float")
    .Attr("num_groups: int")
    .Attr("dtype: {float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto in_shape = c->input(0);
      auto gamma_beta_shape = c->input(1);
      c->set_output(0, in_shape);
      c->set_output(1, gamma_beta_shape);
      c->set_output(2, gamma_beta_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnGroupNormTraining.
)doc");

REGISTER_OP("PopnnGroupNormStatistics")
    .Input("inputs: dtype")
    .Output("mean: dtype")
    .Output("inv_std_dev: dtype")
    .Attr("data_format: string")
    .Attr("epsilon: float")
    .Attr("num_groups: int")
    .Attr("dtype: {float16, float32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int64 num_groups_time_batch;
      TF_RETURN_IF_ERROR(GetMeanAndVarianceSize(c, num_groups_time_batch));
      shape_inference::DimensionOrConstant doc_num_groups_time_batch(
          num_groups_time_batch);
      auto mean_inv_std_dev_shape = c->MakeShape({doc_num_groups_time_batch});

      c->set_output(0, mean_inv_std_dev_shape);
      c->set_output(1, mean_inv_std_dev_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnGroupNormStatistics.
)doc");

}  // namespace tensorflow
