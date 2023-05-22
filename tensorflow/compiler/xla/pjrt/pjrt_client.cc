/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla {

PjRtBuffer::ExternalReference::~ExternalReference() = default;

StatusOr<std::uintptr_t> PjRtClient::UnsafeBufferPointer(PjRtBuffer* buffer) {
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

PjRtFuture<Status> PjRtBuffer::CopyRawToHostFuture(
    PjRtFuture<StatusOr<void*>> dst, int64_t offset, int64_t transfer_size) {
  StatusOr<void*> awaited_dst = dst.Await();
  if (!awaited_dst.ok()) {
    return PjRtFuture<Status>(std::move(awaited_dst).status());
  }
  return CopyRawToHost(*awaited_dst, offset, transfer_size);
}

MultiSliceConfig::~MultiSliceConfig() {}

std::string CompiledMemoryStats::DebugString() const {
  return absl::Substitute(
      "CompiledMemoryStats("
      "generated_code_size_in_bytes=$0, "
      "argument_size_in_bytes=$1, "
      "output_size_in_bytes=$2, "
      "alias_size_in_bytes=$3, "
      "temp_size_in_bytes=$4)",
      generated_code_size_in_bytes, argument_size_in_bytes,
      output_size_in_bytes, alias_size_in_bytes, temp_size_in_bytes);
}

// Defining the first virtual non-pure method, which is usually the virtual
// destructor, makes it a key function. This reduces the program size and takes
// fewer linker resources.
PjRtHostMemoryForDeviceManager::~PjRtHostMemoryForDeviceManager() = default;

CopyToDeviceStream::~CopyToDeviceStream() = default;

StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
PjRtLoadedExecutable::GetCostAnalysis() const {
  // Get HLO cost analysis first
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloCostAnalysis> hlo_cost_analysis,
                      client()->GetHloCostAnalysis());

  // Call into HLO module to accept the analysis, which also calculates the
  // cost properties
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> modules,
                      GetHloModules());
  if (modules.empty()) {
    return NotFound(
        "Executable '%s' did not have an HloModule to generate "
        "cost analysis with.",
        name());
  } else if (modules.size() > 1) {
    return Unimplemented(
        "GetCostAnalysis() doesn't support multiple program "
        "multiple data executables.");
  }

  TF_RETURN_IF_ERROR(
      modules[0]->entry_computation()->Accept(hlo_cost_analysis.get()));

  // Return cost properties
  absl::flat_hash_map<std::string, PjRtValueType> ret;
  hlo_cost_analysis->properties().ForEach(
      [&](absl::string_view key, float val) { ret[key] = val; });
  return ret;
}

}  // namespace xla
