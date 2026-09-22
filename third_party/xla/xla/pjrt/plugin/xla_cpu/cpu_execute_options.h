/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PLUGIN_XLA_CPU_CPU_EXECUTE_OPTIONS_H_
#define XLA_PJRT_PLUGIN_XLA_CPU_CPU_EXECUTE_OPTIONS_H_

#include <optional>

#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/pjrt/pjrt_executable.h"

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace xla {

// ExecuteContext for XLA:CPU PjRtLoadedExecutable::Execute calls.
class CpuExecuteContext : public ExecuteContext {
 public:
  ~CpuExecuteContext() override = default;

  // If specified, override the process ID specified in
  // `CpuClientOptions::process_id` for a particular call of
  // PjRtLoadedExecutable::Execute.
  //
  // TODO(hyeontaek): Look for a collectives-agnostic way and combine this
  // option with `ExecuteOptions::multi_slice_config`.
  std::optional<int>& process_index() { return process_index_; }
  std::optional<int> process_index() const { return process_index_; }

  // If specified, override CPU collectives specified in
  // `CpuClientOptions::collectives` for a particular call of
  // PjRtLoadedExecutable::Execute. Must remain valid until the execution
  // finishes.
  //
  // TODO(hyeontaek): Look for a collectives-agnostic way and combine this
  // option with `ExecuteOptions::multi_slice_config`.
  cpu::CpuCollectives*& collectives() { return collectives_; }
  cpu::CpuCollectives* collectives() const { return collectives_; }

  // If specified, override the intra-op thread pool for a particular call of
  // PjRtLoadedExecutable::Execute. Must remain valid until the execution
  // finishes.
  const Eigen::ThreadPoolDevice* intra_op_thread_pool() const {
    return intra_op_thread_pool_;
  }
  void set_intra_op_thread_pool(
      const Eigen::ThreadPoolDevice* intra_op_thread_pool) {
    intra_op_thread_pool_ = intra_op_thread_pool;
  }

 private:
  std::optional<int> process_index_;
  cpu::CpuCollectives* collectives_ = nullptr;
  const Eigen::ThreadPoolDevice* intra_op_thread_pool_ = nullptr;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_CPU_CPU_EXECUTE_OPTIONS_H_
