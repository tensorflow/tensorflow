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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTOTUNER_COMPILE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTOTUNER_COMPILE_UTIL_H_

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/compiler/xla/autotune_results.pb.h"
#include "tensorflow/compiler/xla/autotuning.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_clone_context.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/autotuner_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

// Autotuning utils which require compiling fusions separately. Requires a
// separate target, as runtime autotuning cannot perform compilation.
class AutotunerCompileUtil {
 public:
  using ExtractModuleFn =
      absl::AnyInvocable<StatusOr<std::unique_ptr<HloModule>>()>;

  static StatusOr<AutotunerCompileUtil> Create(
      se::Stream& stream, se::DeviceMemoryAllocator& allocator);

  AutotunerCompileUtil(Compiler* compiler, se::StreamExecutor& stream_executor,
                       se::Stream& stream, se::DeviceMemoryAllocator& allocator)
      : compiler_(compiler),
        stream_executor_(stream_executor),
        stream_(stream),
        allocator_(allocator) {}

  // Runs the compiled executable with the given extractor, cached with
  // <cache_key, config>. Returns std::nullopt on expected failure, bad Status
  // otherwise.
  // Uses a global cache, *not* unique per instance.
  StatusOr<std::optional<absl::Duration>> GenerateAndProfileExecutable(
      const HloComputation& hlo_computation, const AutotuneResult& config,
      const AutotuneCacheKey& cache_key, se::Stream* stream,
      absl::Span<se::DeviceMemoryBase const> input_buffers,
      se::DeviceMemoryBase output_buffer, ExtractModuleFn extractor);

  // Generic method to compile a given computation in isolation using a given
  // pipeline, cached on AutotuneResult and AutotuneCacheKey.
  //
  // On *expected* failures we will store an empty unique_ptr in cache.
  //
  // Returns:
  //  - <nullptr> on *expected* failure
  //  - Executable if everything goes fine.
  //  - Status on *unexpected* failure.
  StatusOr<Executable*> Compile(
      const HloComputation& hlo_computation, const AutotuneResult& res,
      const AutotuneCacheKey& cache_key,
      AutotunerCompileUtil::ExtractModuleFn extractor);

  static void ClearCompilationCache();

 private:
  StatusOr<std::unique_ptr<Executable>> RunBackend(
      const HloComputation& original_computation,
      std::unique_ptr<HloModule> module);

  StatusOr<std::unique_ptr<Executable>> CompileNoCache(
      const HloComputation& original_computation,
      AutotunerCompileUtil::ExtractModuleFn module_extractor);

  StatusOr<ExecutionOutput> Execute(Executable& executable,
                                    std::vector<ExecutionInput> arguments);

  Compiler* compiler_;
  se::StreamExecutor& stream_executor_;
  se::Stream& stream_;
  se::DeviceMemoryAllocator& allocator_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTOTUNER_COMPILE_UTIL_H_
