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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NVPTX_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NVPTX_EXECUTABLE_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// GPU-targeting implementation of the XLA Executable interface.
//
// Launches the given GPU kernel via the StreamExecutor.
//
// This is an immutable data type after initialization, and thus thread safe.
class NVPTXExecutable : public GpuExecutable {
 public:
  // binary (i.e. the compiled text) may be empty, in which case we leave
  // compilation up to the GPU driver.
  NVPTXExecutable(const string& text, const std::vector<uint8>& binary,
                  std::pair<int, int> compute_capability,
                  std::unique_ptr<const ThunkSchedule> thunk_schedule,
                  std::unique_ptr<const HloModule> hlo_module,
                  std::unique_ptr<const BufferAssignment> assignment,
                  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                  std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map);

  Status CheckCompatibilityWithServiceExecutableRunOptions(
      const ServiceExecutableRunOptions* run_options) override;

 private:
  // The compute capability of the GPU we're targeting with this GpuExecutable.
  std::pair<int, int> compute_capability_;

  TF_DISALLOW_COPY_AND_ASSIGN(NVPTXExecutable);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NVPTX_EXECUTABLE_H_
