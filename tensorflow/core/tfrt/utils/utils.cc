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
#include "tensorflow/core/tfrt/utils/utils.h"

#include <utility>

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/tfrt/eager/virtual_device.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tensorflow/core/tpu/virtual_device.h"
#include "tfrt/bef/bef_encoding.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tfrt {

using ::tensorflow::StatusOr;

Expected<const char*> ConvertTfDeviceNameToTfrt(
    const char* device_name, tensorflow::EagerContext* eager_context) {
  // NOTE(fishx): We need to get tf_device first because DeviceMgr in current TF
  // allows us get the device with simplified name like "CPU:0". However, TFRT
  // DeviceManager only allows get device via its fullname.
  tensorflow::Device* tf_device;
  tensorflow::Status s =
      eager_context->FindDeviceFromName(device_name, &tf_device);
  if (!s.ok()) {
    return MakeStringError(s.error_message());
  }
  return tf_device->name().c_str();
}

DType ConvertTfDTypeToTfrtDType(tensorflow::DataType dtype) {
  switch (dtype) {
#define DTYPE(TFRT_DTYPE, TF_DTYPE) \
  case tensorflow::TF_DTYPE:        \
    return DType(DType::TFRT_DTYPE);
#include "tensorflow/core/tfrt/utils/dtype.def"  // NOLINT
    default:
      return DType();
  }
}

tensorflow::Status RunRuntimeInitializer(const tfrt::ExecutionContext& exec_ctx,
                                         tfrt::BEFFile* bef_file,
                                         absl::string_view fallback_init_func) {
  auto* host = exec_ctx.host();

  auto* func = bef_file->GetFunction(
      {fallback_init_func.data(), fallback_init_func.size()});
  if (func == nullptr) return ::tensorflow::OkStatus();

  if (func->function_kind() == FunctionKind::kBEFFunction) {
    auto ready_chain = GetReadyChain();

    DCHECK_EQ(func->argument_types().size(), 1);

    llvm::SmallVector<RCReference<AsyncValue>, 1> results;
    results.resize(func->result_types().size());
    DCHECK_EQ(results.size(), 1);

    func->Execute(exec_ctx, ready_chain.GetAsyncValue(), results);

    host->Await(results);

    if (auto* error = results[0]->GetErrorIfPresent()) {
      return CreateTfErrorStatus(*error);
    }
  } else {
    DCHECK_EQ(func->result_types().size(), 0);
    if (auto err = ExecuteSyncBEFFunction(*func, exec_ctx, {}, {})) {
      return tensorflow::errors::Internal(
          tfrt::StrCat("Failed to run function: ", func->name(), err));
    }
  }

  return ::tensorflow::OkStatus();
}

void CreateDummyTfDevices(
    const std::vector<std::string>& device_names,
    std::vector<std::unique_ptr<tensorflow::Device>>* dummy_tf_devices) {
  for (const auto& name : device_names) {
    tensorflow::DeviceAttributes device_attrs =
        tensorflow::Device::BuildDeviceAttributes(
            name, tensorflow::DEVICE_TPU_SYSTEM, tensorflow::Bytes(16ULL << 30),
            tensorflow::DeviceLocality(), "device: TFRT TPU SYSTEM device");
    dummy_tf_devices->push_back(std::make_unique<tensorflow::VirtualDevice>(
        tensorflow::Env::Default(), device_attrs));
  }
}

void AddDummyTfrtDevices(const std::vector<std::string>& device_names,
                         HostContext* host_ctx) {
  for (const auto& name : device_names) {
    host_ctx->GetDeviceManager()->MaybeAddDevice(
        TakeRef(new tfrt::VirtualDevice(name)));
  }
}

StatusOr<RCReference<tfrt::BEFFile>> CreateBefFileFromBefBuffer(
    const tensorflow::tfrt_stub::Runtime& runtime, const tfrt::BefBuffer& bef) {
  auto* core_runtime = runtime.core_runtime();
  DCHECK(core_runtime);
  auto* host_context = core_runtime->GetHostContext();
  DCHECK(host_context);
  auto bef_file =
      BEFFile::Open(bef, host_context->GetKernelRegistry(),
                    host_context->diag_handler(), host_context->allocator());
  TF_RET_CHECK(bef_file) << "failed to open BEF";
  return bef_file;
}

int64_t GetUniqueInt() {
  static std::atomic<int64_t> id(0);
  return id.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace tfrt
