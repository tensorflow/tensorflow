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

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

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
  bool has_shape_stats = false;
  std::vector<std::string> input_tensors;
  event.ForEachStat([&](const XStatVisitor& stat) {
    if (stat.Type() == StatType::kLevel0) {
      tf_op = ParseTfOpFullname(stat.StrValue());
    } else if (stat.Type() == StatType::kTensorShapes) {
      has_shape_stats = true;
      auto shapes_stats = stat.StrValue();
      absl::ConsumePrefix(&shapes_stats, "(");
      absl::ConsumeSuffix(&shapes_stats, ")");
      input_tensors = absl::StrSplit(shapes_stats, ';');
    }
  });

  // Return empty OpRoofLineStats if shape is not traced or this is not a tf op.
  if (tf_op.type.empty() || !has_shape_stats) {
    return {0ULL, 0ULL, /*inaccurate=*/true};
  }

  grappler::OpContext op_context;
  op_context.name = std::string(tf_op.type);
  op_context.op_info.set_op(op_context.name);
  for (const auto& tensor : input_tensors) {
    *op_context.op_info.add_inputs() = GetTensorProperties(tensor);
  }
  grappler::Costs costs = PredictCosts(op_context);
  if (costs.inaccurate) unsupported_ops_.insert(std::string(tf_op.type));

  VLOG(1) << tf_op.type << "[" << absl::StrJoin(input_tensors, ",") << "]"
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
