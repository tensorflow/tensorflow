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

#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace tpu {

absl::Status LoadOpShapeFunction::operator()(
    shape_inference::InferenceContext* c) const {
  int table_id;
  TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
  std::string table_name;
  TF_RETURN_IF_ERROR(c->GetAttr("table_name", &table_name));
  // Exactly one must be non-default.
  if ((table_id >= 0) == (!table_name.empty())) {
    return absl::InvalidArgumentError(
        "exactly one of table_id or table_name must be non-default");
  }
  int num_shards;
  TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
  int shard_id;
  TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));

  // Verify shapes have rank 2 and are compatible when they are
  // required to be valid.
  shape_inference::ShapeHandle parameter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &parameter_shape));
  for (int j = 1; j < c->num_inputs(); ++j) {
    shape_inference::ShapeHandle accumulator_j_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(j), 2, &accumulator_j_shape));
    shape_inference::ShapeHandle merged;
    TF_RETURN_IF_ERROR(c->Merge(parameter_shape, accumulator_j_shape, &merged));
  }

  return absl::OkStatus();
}

absl::Status RetrieveOpShapeFunction::operator()(
    shape_inference::InferenceContext* c) const {
  int table_id;
  TF_RETURN_IF_ERROR(c->GetAttr("table_id", &table_id));
  std::string table_name;
  TF_RETURN_IF_ERROR(c->GetAttr("table_name", &table_name));
  // Exactly one must be non-default.
  if ((table_id >= 0) == (!table_name.empty())) {
    return absl::InvalidArgumentError(
        "exactly one of table_id or table_name must be non-default");
  }
  int num_shards;
  TF_RETURN_IF_ERROR(c->GetAttr("num_shards", &num_shards));
  int shard_id;
  TF_RETURN_IF_ERROR(c->GetAttr("shard_id", &shard_id));
  for (int j = 0; j < c->num_outputs(); ++j) {
    c->set_output(j, c->MakeShape(std::vector<shape_inference::DimensionHandle>(
                         2, c->UnknownDim())));
  }
  return absl::OkStatus();
}

}  // namespace tpu
}  // namespace tensorflow
