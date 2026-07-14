/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_THUNKS_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_THUNKS_H_

#include <cassert>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/executable_run_options.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/tsl/platform/threadpool.h"

namespace tensorflow {

class XlaCompiledCpuFunctionThunks : public XlaCompiledCpuFunction {
 public:
  explicit XlaCompiledCpuFunctionThunks(
      const StaticData& static_data,
      AllocMode alloc_mode =
          AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS);

  bool Run() override;

  bool is_thunk_mode() const override { return true; }

  void set_thread_pool(const Eigen::ThreadPoolDevice* pool) override {
    thunk_run_options_.set_intra_op_thread_pool(pool);
  }

 protected:
  std::vector<xla::cpu::NanoRtExecutable::Argument> GenerateNanortArgs();
  std::vector<xla::cpu::NanoRtExecutable::Result> GenerateNanortResults();
  xla::cpu::NanoRtExecutable::PreallocatedTemp GenerateNanortPreallocatedTemp();

 private:
  // For NanoRtExecutable.
  absl::Status Execute(
      absl::Span<const xla::cpu::NanoRtExecutable::Argument> arguments,
      absl::Span<const xla::cpu::NanoRtExecutable::Result> results,
      xla::cpu::NanoRtExecutable::PreallocatedTemp temp);

  std::unique_ptr<xla::cpu::NanoRtExecutable> executable_;
  xla::cpu::NanoRtExecutable::ExecuteOptions thunk_run_options_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_THUNKS_H_
