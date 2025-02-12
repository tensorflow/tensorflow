/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/next_pluggable_device/tensor_pjrt_buffer_util.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/jit/pjrt_tensor_buffer_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"
#include "tensorflow/core/tfrt/common/global_state.h"
#include "tensorflow/core/tfrt/common/pjrt_state.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"

namespace tensorflow {

absl::StatusOr<PJRT_Buffer*> GetPjRtCBufferFromTensor(const Tensor* tensor) {
  tensorflow::AsyncValueTensor* av_tensor =
      tensorflow::AsyncValueTensor::FromTensor(tensor);
  if (av_tensor == nullptr || av_tensor->GetBuffer() == nullptr) {
    return absl::InternalError("Input tensor does not have PjRtBuffer.");
  }
  auto* c_api_buffer =
      dynamic_cast<xla::PjRtCApiBuffer*>(av_tensor->GetBuffer().get());
  if (c_api_buffer == nullptr) {
    return absl::InternalError(
        "The PjRtBuffer in the tensor is not type PjRtCApiBuffer.");
  }
  return c_api_buffer->c_buffer();
}

absl::Status SetPjRtCBufferToTensor(PJRT_Buffer* c_buffer,
                                    xla::PjRtCApiClient* c_api_client,
                                    Tensor* tensor) {
  auto buffer = std::make_unique<xla::PjRtCApiBuffer>(c_api_client, c_buffer);
  tensorflow::AsyncValueTensor* av_tensor =
      tensorflow::AsyncValueTensor::FromTensor(tensor);
  if (av_tensor == nullptr) {
    TF_ASSIGN_OR_RETURN(
        *tensor, MakeTensorFromPjRtBuffer(tensor->dtype(), tensor->shape(),
                                          std::move(buffer)));
  } else {
    av_tensor->SetBuffer(std::move(buffer));
  }
  return absl::OkStatus();
}

absl::StatusOr<xla::PjRtCApiClient*> GetPjRtCApiClient(
    const DeviceType& device_type) {
  TF_ASSIGN_OR_RETURN(absl::StatusOr<xla::PjRtClient*> pjrt_client,
                      tensorflow::GetPjRtClient(device_type));
  auto* pjrt_c_api_client = dynamic_cast<xla::PjRtCApiClient*>(*pjrt_client);
  if (pjrt_c_api_client == nullptr) {
    return absl::InternalError(absl::StrCat("PjRtClient for ",
                                            device_type.type_string(),
                                            " is not type PjRtCApiClient"));
  }
  return pjrt_c_api_client;
}

absl::Status ResetPjRtClient(const DeviceType& device_type) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  PjRtState* pjrt_state;
  TF_RETURN_IF_ERROR(rmgr->Lookup(rmgr->default_container(),
                                  kPjRtStateResourceName, &pjrt_state));
  TF_RETURN_IF_ERROR(pjrt_state->MovePjRtClientToUnused(device_type));
  return absl::OkStatus();
}

}  // namespace tensorflow
