/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_MODULE_GLOBALS_H_
#define XLA_SERVICE_GPU_GPU_MODULE_GLOBALS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

class HloInstruction;

namespace gpu {

class GpuModuleGlobals {
 public:
  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceAddressBase>;

  struct ConstantInfo {
    std::string symbol_name;
    DenseDataIntermediate content;
    int allocation_index = -1;

    GpuExecutableProto::ConstantInfoProto ToProto(
        bool skip_content_serialization = false) const;

    static absl::StatusOr<ConstantInfo> FromProto(
        const GpuExecutableProto::ConstantInfoProto& proto,
        const absl::flat_hash_map<std::string,
                                  const HloInstruction*>* absl_nullable
            content_overrides = nullptr);
  };

  GpuModuleGlobals(const std::vector<uint8_t>& binary,
                   const std::vector<ConstantInfo>& constants)
      : binary_(binary), constants_(constants) {}

  // Loads the executable module for `stream` and initializes constant globals.
  // Loaded modules and resolved global addresses are cached per executor.
  absl::StatusOr<const BufferAllocToDeviceMemoryMap*> Resolve(
      se::Stream* stream);

 private:
  const std::vector<uint8_t>& binary_;
  const std::vector<ConstantInfo>& constants_;

  absl::Mutex mutex_;
  // Cache of module handles. Required to keep loaded modules alive until this
  // helper is destroyed.
  absl::flat_hash_map<se::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ ABSL_GUARDED_BY(mutex_);
  // Cache of constant buffer allocation maps used by `Resolve`.
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<BufferAllocToDeviceMemoryMap>>
      globals_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_MODULE_GLOBALS_H_
