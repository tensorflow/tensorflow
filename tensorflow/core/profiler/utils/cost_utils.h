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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_COST_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_COST_UTILS_H_

#include <set>

#include "absl/strings/string_view.h"
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// This is a wrapper of tensorflow::grappler::OpLevelCostEstimator and use
// tracing time information to estimate the roof line stats for each traced
// tensorflow op.
class TfOpRoofLineCostEstimator
    : public tensorflow::grappler::OpLevelCostEstimator {
 public:
  TfOpRoofLineCostEstimator() = default;
  ~TfOpRoofLineCostEstimator() override;

  grappler::DeviceInfo GetDeviceInfo(
      const DeviceProperties& device) const override;

  struct OpRoofLineStats {
    uint64 flops = 0LL;
    uint64 bytes_accessed = 0LL;
    bool inaccurate = false;
  };
  OpRoofLineStats Predict(const XEventVisitor& event);

 private:
  std::set<string> unsupported_ops_;  // summary for unsupported ops.

  TF_DISALLOW_COPY_AND_ASSIGN(TfOpRoofLineCostEstimator);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_COST_UTILS_H_
