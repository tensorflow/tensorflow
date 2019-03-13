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
      return Status::OK();
    })
    .Doc(R"doc(
Internal implementation of PopnnGroupNormTraining.
)doc");

REGISTER_OP("PopnnGroupNormGrad")
    .Input("inputs: dtype")
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
