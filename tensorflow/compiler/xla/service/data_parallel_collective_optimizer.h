/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DATA_PARALLEL_COLLECTIVE_OPTIMIZER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DATA_PARALLEL_COLLECTIVE_OPTIMIZER_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// This transformation peels off loop iterations of models with stacked layers
// that perform data parallelism using reduce-scatter/all-reduce/all-gather.
// Collective instructions are pushed to the next iteration in which they can
// overlap with the entirely of the next layer rather than with a more limited
// amount of computation in the current iteration. An example of transformation
// is this:
//
// while (i < LAYERS) {
//   p0 = param(0)
//   p1 = param(1)
//   x = computation(p0)
//   xg = all-reduce(x)
//   y = computation(p1)
//   yg = all-reduce(y)
// }
//
// to
//
// x_prev = computation(p0)
// y_prev = computation(p1)
// i = i + 1
// while (i < LAYERS, x_prev, y_prev) {
//   p0 = param(0)
//   p1 = param(1)
//   xg = all-reduce(x_prev)
//   yg = all-reduce(y_prev)
//   x = computation(p0)
//   y = computation(p1)
//   x_prev = x
//   y_prev = y
// }
class DataParallelCollectiveOptimizer : public HloModulePass {
 public:
  enum PipeliningDirection {
    kBackward,
    kForward,
  };
  struct DataParallelCollectiveConfig {
    int64_t level_to_operate_on = 0;
    // Maximum number of HLOs to pipeline per loop. (Meant to help controlling
    // memory pressure manually).
    int64_t max_pipelining_per_loop = 0;
    bool last_run = true;
    bool process_different_sized_ops = false;
    PipeliningDirection pipelining_direction = PipeliningDirection::kForward;
    HloPredicate should_process;
  };
  static const char* const kInsertedByPreviousStep;
  explicit DataParallelCollectiveOptimizer(
      int64_t level_to_operate_on = 0, bool last_run = true,
      bool process_different_sized_ops = false,
      PipeliningDirection pipelining_direction = kForward,
      HloPredicate should_process = nullptr)
      : next_channel_id_(0),
        level_to_operate_on_(level_to_operate_on),
        max_pipelining_per_loop_(INT64_MAX),
        last_run_(last_run),
        process_different_sized_ops_(process_different_sized_ops),
        pipelining_direction_(pipelining_direction),
        should_process_(should_process) {}
  explicit DataParallelCollectiveOptimizer(
      const DataParallelCollectiveConfig& config)
      : next_channel_id_(0),
        level_to_operate_on_(config.level_to_operate_on),
        max_pipelining_per_loop_(config.max_pipelining_per_loop),
        last_run_(config.last_run),
        process_different_sized_ops_(config.process_different_sized_ops),
        pipelining_direction_(config.pipelining_direction),
        should_process_(config.should_process) {}
  DataParallelCollectiveOptimizer(DataParallelCollectiveOptimizer&& other) =
      default;
  DataParallelCollectiveOptimizer& operator=(
      DataParallelCollectiveOptimizer&& other) = default;

  absl::string_view name() const override {
    if (pipelining_direction_ == kForward) {
      return "data-parallel-collective-optimizer-forward";
    } else {
      return "data-parallel-collective-optimizer-backward";
    }
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t next_channel_id_;
  int64_t level_to_operate_on_;
  int64_t max_pipelining_per_loop_;
  bool last_run_;
  bool process_different_sized_ops_;
  PipeliningDirection pipelining_direction_;
  HloPredicate should_process_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DATA_PARALLEL_COLLECTIVE_OPTIMIZER_H_
