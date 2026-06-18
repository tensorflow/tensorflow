/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "xla/tpu/status_helper.h"
#include "xla/tpu/tpu_api.h"
#include "xla/tpu/tpu_ops_c_api.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/dtensor/cc/tpu_system_interface.h"

// Timeout for waiting for TPU devices to appear.
const absl::Duration dtensor_tpu_init_retry_timeout = absl::Seconds(30);

namespace tensorflow {
namespace dtensor {

// Attempt to delete resource_name from resource_manager's default_container.
// Returns OK if the deletion succeeded, or if the resource was not found. Else
// return the deletion error.
template <class ResourceT>
absl::Status DeleteIfExists(ResourceMgr* resource_manager,
                            const char* resource_name) {
  VLOG(1) << "Removing resource " << resource_name << " if it exists";
  absl::Status status = resource_manager->Delete<ResourceT>(
      resource_manager->default_container(), resource_name);
  if (status.ok()) {
    VLOG(1) << "Removed existing resource " << resource_name;
    return absl::OkStatus();
  }
  if (status.code() == error::NOT_FOUND) {
    VLOG(1) << "No resource " << resource_name << " to remove";
    return absl::OkStatus();
  }
  VLOG(1) << "Error removing resource " << resource_name << " : " << status;
  return status;
}

class ConfigureAndInitializeGlobalTPUOpKernel : public OpKernel {
 public:
  explicit ConfigureAndInitializeGlobalTPUOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    LOG(INFO) << "ConfigureAndInitializeGlobalTPUOpKernel op";

    ResourceMgr* rmgr = GetTPUConfigResourceMgr();
    std::vector<int32_t> core_id_output_vec;
    auto retry_timeout = dtensor_tpu_init_retry_timeout;

    VLOG(1) << "Initializing the TPU system.";
    TpuSystemInterface* tpu_system = GetPreferredTpuSystem();
    OP_REQUIRES(ctx, tpu_system != nullptr,
                absl::FailedPreconditionError("TPU system not initialized"));
    OP_REQUIRES_OK(ctx, tpu_system->Initialize(ctx, rmgr, retry_timeout,
                                               &core_id_output_vec));

    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "core_id_output_vec";
      for (auto i : core_id_output_vec) {
        LOG(INFO) << i;
      }
    }

    // Set output using local core ID vector.
    Tensor* ctx_output;
    auto core_id_output_vec_size = core_id_output_vec.size();
    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_output(
            0, TensorShape({static_cast<long long>(core_id_output_vec_size)}),
            &ctx_output));
    for (size_t i = 0; i < core_id_output_vec_size; ++i) {
      ctx_output->flat<int32_t>()(i) = core_id_output_vec[i];
    }

    LOG(INFO) << "ConfigureAndInitializeGlobalTPUOpKernel done";
  }

  ~ConfigureAndInitializeGlobalTPUOpKernel() override = default;

 private:
  // ConfigureAndInitializeGlobalTPUOpKernel is neither copyable nor movable.
  ConfigureAndInitializeGlobalTPUOpKernel(
      const ConfigureAndInitializeGlobalTPUOpKernel&) = delete;
  ConfigureAndInitializeGlobalTPUOpKernel& operator=(
      const ConfigureAndInitializeGlobalTPUOpKernel&) = delete;
};

class ShutdownTPUSystemOpKernel : public OpKernel {
 public:
  explicit ShutdownTPUSystemOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    LOG(INFO) << "ShutdownTPUSystemOpKernel op";

    absl::Status status;
    TpuSystemInterface* tpu_system = GetPreferredTpuSystem();
    OP_REQUIRES(ctx, tpu_system != nullptr,
                absl::FailedPreconditionError("TPU system not initialized."));
    VLOG(1) << "Shutting down the TPU system.";
    status = tpu_system->Shutdown();

    Tensor* output_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({1}), &output_tensor));

    if (status.ok()) {
      output_tensor->flat<bool>()(0) = true;
    } else {
      output_tensor->flat<bool>()(0) = false;
    }
  }
};

class SetGlobalTPUArrayOpKernel : public OpKernel {
 public:
  explicit SetGlobalTPUArrayOpKernel(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "SetGlobalTPUArrayOpKernel op";
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->input(0).shape()),
                absl::InvalidArgumentError(
                    absl::StrCat("Expected argument 0 to be a scalar. Received",
                                 ctx->input(0).DebugString())));
    auto tpu_topology = ctx->input(0).scalar<tstring>()();

    StatusHelper status;
    stream_executor::tpu::OpsApiFn()->SetGlobalTPUArrayOp_DoWorkFn(
        tpu_topology.size(), tpu_topology.data(), status.c_status);
    OP_REQUIRES_OK(ctx, status.status());

    VLOG(1) << "SetGlobalTPUArrayOpKernel done";
  }
};

REGISTER_KERNEL_BUILDER(Name("ConfigureAndInitializeGlobalTPU")
                            .Device(DEVICE_TPU_SYSTEM)
                            .HostMemory("output"),
                        ConfigureAndInitializeGlobalTPUOpKernel);

REGISTER_KERNEL_BUILDER(Name("ShutdownTPUSystem").Device(DEVICE_TPU_SYSTEM),
                        ShutdownTPUSystemOpKernel);

REGISTER_KERNEL_BUILDER(Name("DTensorSetGlobalTPUArray")
                            .Device(DEVICE_TPU_SYSTEM)
                            .HostMemory("topology"),
                        SetGlobalTPUArrayOpKernel);

}  // namespace dtensor
}  // namespace tensorflow
