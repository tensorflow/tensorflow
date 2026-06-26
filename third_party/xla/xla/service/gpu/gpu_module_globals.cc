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

#include "xla/service/gpu/gpu_module_globals.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/scoped_module_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla::gpu {

GpuExecutableProto::ConstantInfoProto GpuModuleGlobals::ConstantInfo::ToProto(
    bool skip_content_serialization) const {
  GpuExecutableProto::ConstantInfoProto proto;
  proto.set_symbol_name(symbol_name);
  if (!skip_content_serialization) {
    *proto.mutable_content() = content.ToProto();
  }
  proto.set_allocation_index(allocation_index);
  return proto;
}

absl::StatusOr<GpuModuleGlobals::ConstantInfo>
GpuModuleGlobals::ConstantInfo::FromProto(
    const GpuExecutableProto::ConstantInfoProto& proto,
    const absl::flat_hash_map<std::string, const HloInstruction*>* absl_nullable
        content_overrides) {
  if (content_overrides) {
    auto it = content_overrides->find(proto.symbol_name());
    if (it == content_overrides->end()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Instruction for ", proto.symbol_name(), " constant missing."));
    }
    const HloInstruction* instr = it->second;
    const Literal& literal = instr->literal();
    auto base = static_cast<const uint8_t*>(literal.untyped_data());
    return ConstantInfo{proto.symbol_name(),
                        DenseDataIntermediate::Alias(
                            absl::MakeSpan(base, base + literal.size_bytes())),
                        static_cast<int>(proto.allocation_index())};
  }
  return ConstantInfo{proto.symbol_name(),
                      DenseDataIntermediate::FromProto(proto.content()),
                      static_cast<int>(proto.allocation_index())};
}

absl::StatusOr<const GpuModuleGlobals::BufferAllocToDeviceMemoryMap*>
GpuModuleGlobals::Resolve(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  absl::MutexLock lock(mutex_);
  auto it = globals_.find(executor);
  if (it != globals_.end()) {
    return it->second.get();
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary_.empty()) {
    module_spec.AddCudaCubinInMemory(binary_);
  }

  auto globals = std::make_unique<BufferAllocToDeviceMemoryMap>();
  se::ModuleHandle module_handle;
  // The CUDA driver isn't able to load a PTX and a binary which are both empty.
  // It's okay if we skip loading in this case; if the module isn't loaded, all
  // symbol lookups will fail, just as they should for an empty module.
  if (!(executor->GetPlatform()->id() == se::cuda::kCudaPlatformId &&
        binary_.empty())) {
    ASSIGN_OR_RETURN(module_handle, executor->LoadModule(module_spec));
  }

  // A flag signalling if constant initialization submitted memcpy operations
  // to the `stream`.
  int submitted_mem_copies = 0;

  for (const ConstantInfo& info : constants_) {
    absl::StatusOr<se::DeviceAddressBase> global_status;
    if (static_cast<bool>(module_handle)) {
      global_status = executor->GetSymbol(info.symbol_name, module_handle);
    }

    se::DeviceAddressBase global;

    CHECK(static_cast<bool>(module_handle) && global_status.ok());
    // The constant was defined in the PTX and has been allocated by the CUDA
    // driver.
    global = *global_status;
    XLA_VLOG_DEVICE(3, executor->device_ordinal()) << absl::StreamFormat(
        "Resolved global %s to %p", info.symbol_name, global.opaque());

    if (!info.content.span().empty()) {
      // This means the constant did not have an initializer in the PTX and
      // therefore must be initialized by XLA here.
      RETURN_IF_ERROR(stream->Memcpy(&global, info.content.span().data(),
                                     info.content.span().size()));
      submitted_mem_copies = true;
    }

    if (info.allocation_index != -1) {
      InsertOrDie(globals.get(), info.allocation_index, global);
    }
  }

  // Wait for the completion of all host->device transfers, to guarantee that
  // destructor will not race with any operations in flight (deallocate
  // xla::Literal owned by the HLO module).
  if (submitted_mem_copies) {
    CHECK_OK(stream->BlockHostUntilDone());
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return globals_.emplace(executor, std::move(globals)).first->second.get();
}

}  // namespace xla::gpu
