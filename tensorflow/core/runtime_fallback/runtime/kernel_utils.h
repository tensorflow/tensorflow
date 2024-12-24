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

// This file declares kernel utils.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_KERNEL_UTILS_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_KERNEL_UTILS_H_

#include <cassert>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

template <typename T>
struct AutoReleaser {
  void operator()(T* p) const { p->Release(); }
};
template <typename T>
using AutoReleasePtr = std::unique_ptr<T, AutoReleaser<T>>;

using OwnedEagerContext = AutoReleasePtr<EagerContext>;
using OwnedEagerOperation = AutoReleasePtr<EagerOperation>;
using OwnedTensorHandle = AutoReleasePtr<TensorHandle>;
using OwnedAbstractTensorInterface = AutoReleasePtr<AbstractTensorInterface>;

// Check if a TensorHandle physically resides on GPU.
inline bool IsGpuTensorHandle(const tensorflow::TensorHandle& handle) {
  absl::Status dummy_status;
  // BackingDeviceName is where the tensor is physically located, not where the
  // op that produces the tensor is.
  // Note that dummy_status is never set in TensorHandle::BackingDeviceName.
  absl::string_view device_name = handle.BackingDeviceName(&dummy_status);
  return absl::StrContains(device_name, "GPU");
}

// TODO(zhangqiaorjc): Allowlist more dtypes as tfrt GPU supports more.
// RuntimeFallbackTensor of supported dtypes below will be eagerly converted to
// tfrt::DenseGpuTensor after each RuntimeFallbackOpHandler::Execute.
inline bool IsSupportedByTFRTGpu(DataType dtype) {
  switch (dtype) {
    default:
      return false;
    case DataType::DT_FLOAT:
    case DataType::DT_DOUBLE:
    case DataType::DT_INT32:
      return true;
  }
}

// TODO(b/165872892): Remove this method.
// This method is needed because we use different device name in TF-TFRT
// integration and mlir test. In TF-TFRT integration, we reuse the device full
// name (e.g. /job:localhost/replica:0/task:0/device:GPU:0) from TF. But in mlir
// test, we use simplified device name "GPU:0". And lot of things in fallback
// need to be used in both cases. As a result, we need to look up the device
// with both device names.
inline const char* ConvertTfDeviceNameToTfrtDefault(const char* device_name) {
  assert(strlen(device_name) >= 5);
  return &device_name[strlen(device_name) - 5];
}

// Create and initialize EagerContext.
tfrt::Expected<OwnedEagerContext> InitEagerContext();

tfrt::Expected<OwnedEagerContext> InitEagerContext(
    DynamicDeviceMgr* device_mgr, const SessionOptions& session_opts,
    ContextDevicePlacementPolicy default_device_placement_policy,
    bool is_async);

// Obtain EagerContext from ExecutionContext.
tfrt::Expected<EagerContext*> GetEagerContext(tfrt::ExecutionContext exec_ctx);

// Return the CoreRuntimeOp for `op_name` using fallback op_handler.
llvm::Expected<tfrt::CoreRuntimeOp> GetFallbackOp(tfrt::string_view op_name,
                                                  tfrt::HostContext* host);

constexpr char kEagerContextResourceName[] = "EagerContextResourceName";

class EagerContextResource {
 public:
  explicit EagerContextResource()
      : device_mgr_(std::make_unique<DynamicDeviceMgr>()),
        ctx_{InitEagerContext(
            device_mgr_.get(), tensorflow::SessionOptions(),
            tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
            /*is_async=*/false)} {}
  explicit EagerContextResource(
      const SessionOptions& session_opts,
      ContextDevicePlacementPolicy default_device_placement_policy,
      bool is_async)
      : device_mgr_(std::make_unique<DynamicDeviceMgr>()),
        ctx_{InitEagerContext(device_mgr_.get(), session_opts,
                              default_device_placement_policy, is_async)} {}

  tfrt::Expected<EagerContext*> GetTFEagerContext() {
    if (!ctx_) return ctx_.takeError();
    return ctx_.get().get();
  }

  DynamicDeviceMgr* GetDeviceMgr() { return device_mgr_.get(); }

  llvm::Error AddDevices(std::vector<std::unique_ptr<Device>> devices) {
    if (!ctx_) return ctx_.takeError();
    absl::Status s = dynamic_cast<tensorflow::DynamicDeviceMgr*>(
                         ctx_.get()->local_device_mgr())
                         ->AddDevices(std::move(devices));
    if (!s.ok()) return tfrt::MakeStringError(s.message());
    ctx_.get()->InitPrioritizedDeviceTypeList();
    ctx_.get()->pflr()->InitializeDeviceAndFlr();
    return llvm::Error::success();
  }

 private:
  // EagerContext uses this device_mgs as local_device_mgr. We manage the
  // device_mgr_ here to allow TFRT adding new devices after EagerContext
  // initialization.
  // Today, TFRT only adds TPU devices after EagerContext initialization.
  std::unique_ptr<DynamicDeviceMgr> device_mgr_;

  tfrt::Expected<OwnedEagerContext> ctx_;
};

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_KERNEL_UTILS_H_
