/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/gpu_event_stats.h"

#include "absl/strings/str_split.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kAnnotationDelimiter = "::";

}

GpuEventStats::GpuEventStats(const XEventVisitor* event) {
  event->ForEachStat([&](const XStatVisitor& stat) {
    if (!stat.Type().has_value()) return;
    switch (stat.Type().value()) {
      case StatType::kTfOp:
        tf_op_fullname = stat.StrOrRefValue();
        break;
      case StatType::kEquation:
        equation = stat.StrOrRefValue();
        break;
      case StatType::kTensorShapes:
        tensor_shapes = stat.StrOrRefValue();
        break;
      case StatType::kHloOp:
        hlo_op_names =
            absl::StrSplit(stat.StrOrRefValue(), kAnnotationDelimiter);
        break;
      case StatType::kHloModule:
        hlo_module_name = stat.StrOrRefValue();
        break;
      case StatType::kProgramId:
        program_id = stat.IntOrUintValue();
        break;
      case StatType::kKernelDetails:
        kernel_details = stat.StrOrRefValue();
        break;
      case StatType::kMemcpyDetails:
        memcpy_details = stat.StrOrRefValue();
        break;
      case StatType::kCorrelationId:
        correlation_id = stat.IntValue();
        break;
      case StatType::kGroupId:
        group_id = stat.IntValue();
        break;
      case StatType::kIsEager:
        is_eager = stat.BoolValue();
        break;
      default:
        break;
    }
  });
}

}  // namespace profiler
}  // namespace tensorflow
