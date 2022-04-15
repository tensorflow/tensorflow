/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/tpu/tpu_op_executable.h"

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {

TpuOpExecutable::TpuOpExecutable(const XLA_TpuProgram* core_program,
                                 std::unique_ptr<xla::HloModule> hlo_module,
                                 HostCommandHandler host_command_handler)
    : TpuExecutableInterface(std::move(hlo_module)),
      core_program_(core_program),
      host_command_handler_(std::move(host_command_handler)) {}

Status TpuOpExecutable::LoadProgramAndEnqueueToStream(
    const xla::ServiceExecutableRunOptions& run_options,
    absl::Span<const se::DeviceMemoryBase> arguments,
    se::DeviceMemoryBase result,
    absl::optional<se::DeviceMemoryBase> cross_program_prefetch_addr) {
  SE_DeviceMemoryBase* arguments_bases = nullptr;
  if (!arguments.empty()) {
    arguments_bases = new SE_DeviceMemoryBase[arguments.size()];
    for (int i = 0; i < arguments.size(); i++) {
      arguments_bases[i] =
          SE_DeviceMemoryBase{const_cast<void*>(arguments[i].opaque()),
                              arguments[i].size(), arguments[i].payload()};
    }
  }

  SE_DeviceMemoryBase result_base{result.opaque(), result.size(),
                                  result.payload()};
  SE_DeviceMemoryBase prefetch_base;
  if (cross_program_prefetch_addr.has_value()) {
    prefetch_base = SE_DeviceMemoryBase{cross_program_prefetch_addr->opaque(),
                                        cross_program_prefetch_addr->size(),
                                        cross_program_prefetch_addr->payload()};
  }
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
  auto stream = platform->LookupStream(
      run_options.run_options().stream()->implementation());
  StatusHelper status;

  TpuExecutable_LoadProgramAndEnqueueToStream_Params params;
  params.struct_size = TpuExecutable_LoadProgramAndEnqueueToStream_Params_SIZE;
  params.priv = nullptr;
  params.program = core_program_;
  params.arguments = arguments_bases;
  params.arguments_len = arguments.size();
  params.result = &result_base;
  params.has_cross_program_prefetch_addr =
      cross_program_prefetch_addr.has_value();
  params.cross_program_prefetch_addr =
      cross_program_prefetch_addr.has_value() ? &prefetch_base : nullptr;
  params.rng_seed = rng_seed;
  params.device_assignment = &c_dev_assign;
  params.stream = stream;
  params.status = status.c_status;

  tpu::OpsApiFn()->TpuExecutable_LoadProgramAndEnqueueToStreamFn(&params);

  if (dev_assign != nullptr) {
    stream_executor::tpu::SerializedProto_Free(dev_assign_serialized);
  }
  delete[] arguments_bases;
  return status.status();
}

xla::Shape TpuOpExecutable::HostShapeToDeviceShape(
    const xla::Shape& host_shape) {
  XLA_Shape c_host_shape;
  XLA_Shape c_device_shape;
  ApiConverter::ToC(host_shape, &c_host_shape);
  tpu::OpsApiFn()->HardwareLayout_HostShapeToDeviceShapeFn(&c_host_shape,
                                                           &c_device_shape);
  xla::Shape device_shape = ApiConverter::FromC(&c_device_shape);
  ApiConverter::Destroy(&c_host_shape);
  ApiConverter::Destroy(&c_device_shape);
  return device_shape;
}

int64_t TpuOpExecutable::ShapeSize(const xla::Shape& shape) {
  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);
  int64_t size = tpu::OpsApiFn()->HardwareLayout_ShapeSizeFn(&c_shape);
  ApiConverter::Destroy(&c_shape);
  return size;
}

absl::string_view TpuOpExecutable::fingerprint() const {
  // TODO(skye): the fingerprint can be plumbed through via core_program_
  LOG(FATAL) << "TpuOpExecutable::fingerprint() unimplemented";
}

}  // namespace tensorflow
