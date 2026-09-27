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
#include "tensorflow/core/tpu/kernels/tpu_configuration_ops.h"

#include "absl/status/status.h"
#include "xla/tpu/tpu_ops_c_api.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"  // IWYU pragma: keep

namespace tensorflow {

namespace {

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
}  // namespace

void ShutdownDistributedTpuOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "ShutdownDistributedTpuOp";
  XLA_SCOPED_LOGGING_TIMER("ShutdownDistributedTpuOp");

  auto* rmgr = GetTPUConfigResourceMgr();
  OP_REQUIRES_OK(ctx, DeleteIfExists<tpu::TpuMeshStateInterface>(
                          rmgr, tpu::kTpuMeshStateInterfaceResourceName));

  VLOG(1) << "ShutdownDistributedTpuOp done";
}

// These ops execute on the TPU_SYSTEM device only.
REGISTER_KERNEL_BUILDER(
    Name("_ShutdownDistributedTPU").Device(DEVICE_TPU_SYSTEM),
    ShutdownDistributedTpuOp);

}  // namespace tensorflow
