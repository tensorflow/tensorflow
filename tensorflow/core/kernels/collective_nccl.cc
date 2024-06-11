/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/collective_nccl.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/nccl/nccl_manager.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

NcclBase::NcclBase(CollectiveType type, const string& name)
    : type_(type), name_(name), col_ctx_(nullptr), col_params_(nullptr) {}

Status NcclBase::InitializeCollectiveParams(CollectiveParams* col_params) {
  if (type_ != col_params->instance.type) {
    return errors::Internal("Expected initialized type ", type_,
                            " to match type in CollectiveParams ",
                            col_params->instance.type);
  }

  const char* expected_name;
  switch (type_) {
    case REDUCTION_COLLECTIVE:
      expected_name = "NcclReduce";
      break;
    case BROADCAST_COLLECTIVE:
      expected_name = "NcclBroadcast";
      break;
    case GATHER_COLLECTIVE:
      expected_name = "NcclGather";
      break;
    case REDUCE_SCATTER_COLLECTIVE:
      expected_name = "NcclReduceScatter";
      break;
    case ALL_TO_ALL_COLLECTIVE:
      expected_name = "NcclAllToAll";
      break;
    default:
      return errors::Internal("Unexpected CollectiveType ", type_);
  }

  if (expected_name != col_params->instance.impl_details.collective_name) {
    return errors::Internal("Unexpected combination of collective type ",
                            col_params->instance.type, " and collective name ",
                            col_params->instance.impl_details.collective_name,
                            ", expected name ", expected_name);
  }

  return OkStatus();
}

Status NcclBase::InitializeCollectiveContext(
    std::shared_ptr<CollectiveContext> col_ctx) {
  col_ctx_ = col_ctx;
  col_params_ = col_ctx->col_params.get();
  return collective_util::InitializeDeviceAndLocality(
      col_ctx->dev_mgr, col_ctx->device_name, &col_ctx->device,
      &col_ctx->device_locality);
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
