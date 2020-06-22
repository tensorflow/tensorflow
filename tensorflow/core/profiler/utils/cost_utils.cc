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

#include "tensorflow/core/profiler/utils/cost_utils.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_context.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

namespace {

// Decode the string that encodes tensor shape and type information and convert
// to TensorProperties.
// Returns an empty TensorProperties if error or input is "".
// See OpKernel::TraceString() to see when the shape is encoded as "".
// Input format is <DTYPE>[<dim1>, <dim2>,...]
static OpInfo::TensorProperties GetTensorProperties(absl::string_view info) {
  OpInfo::TensorProperties tensor_prop;
  std::vector<absl::string_view> parts = absl::StrSplit(info, '[');
  if (parts.size() != 2) return tensor_prop;
  DataType data_type = DT_INVALID;
  if (!DataTypeFromString(parts[0], &data_type)) return tensor_prop;
  tensor_prop.set_dtype(data_type);
  absl::ConsumeSuffix(&parts[1], "]");
  if (parts[1].empty()) {  // Scalar type.
    tensor_prop.mutable_shape()->add_dim()->set_size(1);
    return tensor_prop;
  }
  std::vector<absl::string_view> dims = absl::StrSplit(parts[1], ',');
  for (const auto dim : dims) {
    int size;
    if (!absl::SimpleAtoi(dim, &size)) return OpInfo::TensorProperties();
    tensor_prop.mutable_shape()->add_dim()->set_size(size);
  }
  return tensor_prop;
}

}  // namespace

TfOpRoofLineCostEstimator::~TfOpRoofLineCostEstimator() {
  if (!unsupported_ops_.empty()) {
    LOG(ERROR) << "Unsupported Op for Roofline Cost Analysis are:"
               << absl::StrJoin(unsupported_ops_, ",");
  }
}

grappler::DeviceInfo TfOpRoofLineCostEstimator::GetDeviceInfo(
    const DeviceProperties& device) const {
  // Hypothetical devices that is used to measure peak flops and memory bytes
  // accessed.
  return grappler::DeviceInfo(/*gigaops=*/1, /*gb_per_sec=*/1);
}

TfOpRoofLineCostEstimator::OpRoofLineStats TfOpRoofLineCostEstimator::Predict(
    const XEventVisitor& event) {
  TfOp tf_op;
  absl::string_view tensor_shapes;
  event.ForEachStat([&](const XStatVisitor& stat) {
    if (!stat.Type().has_value()) return;
    switch (stat.Type().value()) {
      case StatType::kLevel0:
        tf_op = ParseTfOpFullname(stat.StrOrRefValue());
        break;
      case StatType::kTensorShapes:
        tensor_shapes = stat.StrOrRefValue();
        break;
    }
  });

  // Return empty OpRoofLineStats if shape is not traced or this is not a tf op.
  if (tf_op.type.empty() || tensor_shapes.empty()) {
    return {0ULL, 0ULL, /*inaccurate=*/true};
  }

  grappler::OpContext op_context;
  op_context.name = std::string(tf_op.type);
  op_context.op_info.set_op(op_context.name);
  for (absl::string_view tensor : ParseTensorShapes(tensor_shapes)) {
    *op_context.op_info.add_inputs() = GetTensorProperties(tensor);
  }
  grappler::Costs costs = PredictCosts(op_context);
  if (costs.inaccurate) unsupported_ops_.insert(std::string(tf_op.type));

  VLOG(1) << tf_op.type << tensor_shapes
          << " flops:" << costs.compute_time.count()
          << " bytes:" << costs.memory_time.count();

  /* The compute_time is measured in nanoseconds, therefore numerically it is
   * equal to flops because giga ops / second cancel the nanoseconds.
   * Same for memory_time */
  return {/*flops=*/static_cast<uint64>(costs.compute_time.count()),
          /*bytes_accessed=*/static_cast<uint64>(costs.memory_time.count()),
          /*inaccurate=*/costs.inaccurate};
}

}  // namespace profiler
}  // namespace tensorflow
