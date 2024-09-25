/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_client.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/utils.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

PjRtBuffer::ExternalReference::~ExternalReference() = default;

absl::StatusOr<std::uintptr_t> PjRtClient::UnsafeBufferPointer(
    PjRtBuffer* buffer) {
  if (buffer->on_device_shape().IsTuple()) {
    return Unimplemented(
        "unsafe_buffer_pointer is not implemented for tuple buffers.");
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
      buffer->AcquireExternalReference());
  const void* ptr = external_reference_hold->OpaqueDeviceMemoryDataPointer();
  return absl::bit_cast<std::uintptr_t>(ptr);
}

PjRtFuture<> PjRtBuffer::CopyRawToHostFuture(PjRtFuture<void*> dst,
                                             int64_t offset,
                                             int64_t transfer_size) {
  return PjRtFuture<>(absl::UnimplementedError(
      "PjRtBuffer::CopyRawToHostFuture is not implemented"));
}

std::string CompiledMemoryStats::DebugString() const {
  return absl::Substitute(
      "CompiledMemoryStats("
      "generated_code_size_in_bytes=$0, "
      "argument_size_in_bytes=$1, "
      "output_size_in_bytes=$2, "
      "alias_size_in_bytes=$3, "
      "temp_size_in_bytes=$4, "
      "host_generated_code_size_in_bytes=$5, "
      "host_argument_size_in_bytes=$6, "
      "host_output_size_in_bytes=$7, "
      "host_alias_size_in_bytes=$8, "
      "host_temp_size_in_bytes=$9)",
      generated_code_size_in_bytes, argument_size_in_bytes,
      output_size_in_bytes, alias_size_in_bytes, temp_size_in_bytes,
      host_generated_code_size_in_bytes, host_argument_size_in_bytes,
      host_output_size_in_bytes, host_alias_size_in_bytes,
      host_temp_size_in_bytes);
}

// Defining the first virtual non-pure method, which is usually the virtual
// destructor, makes it a key function. This reduces the program size and takes
// fewer linker resources.
PjRtHostMemoryForDeviceManager::~PjRtHostMemoryForDeviceManager() = default;

CopyToDeviceStream::~CopyToDeviceStream() = default;

absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
PjRtLoadedExecutable::GetCostAnalysis() const {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloCostAnalysis> hlo_cost_analysis,
                      client()->GetHloCostAnalysis());
  return PjRtExecutableUtil::RunHloCostAnalysis(*this, hlo_cost_analysis.get());
}

}  // namespace xla
