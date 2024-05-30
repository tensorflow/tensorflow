/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/stream_executor/tpu/tpu_op_executable.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"  // IWYU pragma: keep
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/proto_helper.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_executable_interface.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "xla/stream_executor/tpu/tpu_platform.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

TpuOpExecutable::TpuOpExecutable(
    const XLA_TpuProgram* core_program,
    std::unique_ptr<xla::HloModule> hlo_module,
    SE_OutsideCompilationParams* outside_compilation_params)
    : TpuExecutableInterface(std::move(hlo_module)),
      core_program_(core_program),
      outside_compilation_params_(outside_compilation_params) {}

absl::Status TpuOpExecutable::LoadProgramAndEnqueueToStream(
    const xla::ServiceExecutableRunOptions& run_options,
    absl::Span<const se::DeviceMemoryBase> arguments,
    se::DeviceMemoryBase result,
    const std::vector<se::DeviceMemoryBase>& cross_program_prefetch_addrs,
    const std::vector<uint32_t>& cross_program_prefetch_offsets) {
  auto DeviceMemoryBaseToC = [](const se::DeviceMemoryBase& addr) {
    return SE_DeviceMemoryBase{const_cast<void*>(addr.opaque()), addr.size(),
                               addr.payload()};
  };

  std::vector<SE_DeviceMemoryBase> arguments_bases;
  arguments_bases.resize(arguments.size());
  absl::c_transform(arguments, arguments_bases.begin(), DeviceMemoryBaseToC);

  SE_DeviceMemoryBase result_base = DeviceMemoryBaseToC(result);

  std::vector<SE_DeviceMemoryBase> prefetch_bases;
  prefetch_bases.resize(cross_program_prefetch_addrs.size());
  absl::c_transform(cross_program_prefetch_addrs, prefetch_bases.begin(),
                    DeviceMemoryBaseToC);
  int32_t rng_seed = run_options.run_options().rng_seed();

  XLA_DeviceAssignment c_dev_assign{/*bytes=*/nullptr, /*size=*/0};
  auto dev_assign = run_options.run_options().device_assignment();
  stream_executor::tpu::SerializedProto dev_assign_serialized;
  if (dev_assign != nullptr) {
    xla::DeviceAssignmentProto dev_assign_proto;
    TF_RETURN_IF_ERROR(dev_assign->Serialize(&dev_assign_proto));
    dev_assign_serialized =
        stream_executor::tpu::SerializeProto(dev_assign_proto);
    c_dev_assign.bytes = dev_assign_serialized.bytes;
    c_dev_assign.size = dev_assign_serialized.size;
  }

  auto platform = down_cast<tpu::TpuPlatform*>(
      tpu::TpuPlatformInterface::GetRegisteredPlatform());
  auto stream = platform->LookupStream(run_options.run_options().stream());
  StatusHelper status;

  TpuExecutable_LoadProgramAndEnqueueToStream_Params params;
  params.struct_size = TpuExecutable_LoadProgramAndEnqueueToStream_Params_SIZE;
  params.priv = nullptr;
  params.program = core_program_;
  params.arguments = arguments_bases.empty() ? nullptr : arguments_bases.data();
  params.arguments_len = arguments_bases.size();
  params.result = &result_base;
  params.cross_program_prefetch_addrs =
      prefetch_bases.empty() ? nullptr : prefetch_bases.data();
  params.cross_program_prefetch_addrs_len = prefetch_bases.size();
  params.cross_program_prefetch_offsets =
      cross_program_prefetch_offsets.empty()
          ? nullptr
          : cross_program_prefetch_offsets.data();
  params.cross_program_prefetch_offsets_len =
      cross_program_prefetch_offsets.size();
  params.rng_seed = rng_seed;
  params.device_assignment = &c_dev_assign;
  params.stream = stream;
  params.outside_compilation_params = outside_compilation_params_;
  params.status = status.c_status;

  stream_executor::tpu::OpsApiFn()
      ->TpuExecutable_LoadProgramAndEnqueueToStreamFn(&params);

  if (dev_assign != nullptr) {
    stream_executor::tpu::SerializedProto_Free(dev_assign_serialized);
  }
  return status.status();
}

absl::string_view TpuOpExecutable::fingerprint() const {
  // TODO(skye): the fingerprint can be plumbed through via core_program_
  LOG(FATAL) << "TpuOpExecutable::fingerprint() unimplemented";
}

}  // namespace tensorflow
