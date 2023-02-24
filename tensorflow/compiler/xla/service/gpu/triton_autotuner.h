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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRITON_AUTOTUNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRITON_AUTOTUNER_H_

#include <optional>

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Find best tiling configuration for each triton fusion outlined.
// num_extra_threads: number of threads the pass can use to perform compilation.
// TODO(b/266210099): Use existing thread pool instead?
class TritonAutotuner : public HloModulePass {
 public:
  TritonAutotuner(se::StreamExecutor* stream_exec,
                  se::DeviceMemoryAllocator* allocator,
                  int num_extra_threads = 0)
      : stream_exec_(stream_exec),
        allocator_(allocator),
        num_extra_threads_(num_extra_threads) {}

  absl::string_view name() const override { return "triton-autotuner"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::StreamExecutor* stream_exec_;
  se::DeviceMemoryAllocator* allocator_;
  int num_extra_threads_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRITON_AUTOTUNER_H_
