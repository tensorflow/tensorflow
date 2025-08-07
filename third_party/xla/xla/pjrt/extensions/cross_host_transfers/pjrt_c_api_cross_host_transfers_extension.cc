/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/extensions/cross_host_transfers/pjrt_c_api_cross_host_transfers_extension.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"

namespace pjrt {

namespace {
static xla::PjRtCrossHostRecvNotifier CCrossHostRecvNotifierToCpp(
    const PJRT_Transfers_CrossHostRecvNotifierInfo& c_notifier) {
  return [user_arg = c_notifier.user_arg, notifier = c_notifier.notifier](
             absl::StatusOr<xla::PjRtCrossHostRecvState> recv_state) {
    if (!recv_state.ok()) {
      auto error = new PJRT_Error{recv_state.status()};
      return notifier(error, nullptr, nullptr, 0, user_arg);
    }
    auto& descriptors = recv_state->descriptors;
    std::vector<size_t> descriptors_sizes;
    descriptors_sizes.reserve(descriptors.size());
    std::vector<const char*> serialized_descriptors;
    serialized_descriptors.reserve(descriptors.size());
    for (int i = 0; i < descriptors.size(); ++i) {
      serialized_descriptors.push_back(
          descriptors[i].serialized_descriptors.front().c_str());
      descriptors_sizes.push_back(
          descriptors[i].serialized_descriptors.front().size());
    }
    return notifier(nullptr, serialized_descriptors.data(),
                    descriptors_sizes.data(), descriptors.size(), user_arg);
  };
}
}  // namespace

PJRT_Error* PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers(
    PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_MakeCrossHostReceiveBuffers_Args",
      PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args_STRUCT_SIZE,
      args->struct_size));
  std::vector<xla::Shape> shapes;
  shapes.reserve(args->num_shapes);
  for (int i = 0; i < args->num_shapes; ++i) {
    PJRT_ASSIGN_OR_RETURN(
        xla::Shape shape,
        pjrt::BuildXlaShapeFromC(args->element_types[i], args->num_dims[i],
                                 args->shape_num_dims[i], args->layouts[i]));
    shapes.push_back(std::move(shape));
  }
  xla::PjRtCrossHostRecvNotifier notifier =
      CCrossHostRecvNotifierToCpp(args->notifier);
  PJRT_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers,
      args->client->client->MakeCrossHostReceiveBuffers(
          absl::MakeSpan(shapes), args->device->device, std::move(notifier)));
  args->num_buffers = buffers.size();
  for (int i = 0; i < buffers.size(); ++i) {
    args->buffers[i] = new PJRT_Buffer{std::move(buffers[i]), args->client};
  }
  return nullptr;
}

void PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice(
    PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args* args) {
  std::string serialized_descriptor = std::string(
      args->serialized_descriptor, args->serialized_descriptor_size);
  xla::PjRtFuture<std::string>::Promise promise =
      xla::PjRtFuture<std::string>::CreatePromise();
  promise.Set(std::move(serialized_descriptor));
  auto descriptor_future = xla::PjRtFuture<std::string>(std::move(promise));
  // TODO(emilyaf): Support on_done callback.
  xla::PjRtBuffer::RemoteSendCallback on_done =
      [](absl::Status status, bool sends_were_enqueued) { CHECK_OK(status); };
  args->buffer->buffer->CopyToRemoteDevice(descriptor_future, on_done);
}

PJRT_CrossHostTransfers_Extension CreateCrossHostTransfersExtension(
    PJRT_Extension_Base* next) {
  return PJRT_CrossHostTransfers_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_CrossHostTransfers_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_CrossHostTransfers,
          /*next=*/next,
      },
      /*PJRT_CrossHostTransfers_PJRT_Client_MakeCrossHostReceiveBuffers=*/
      PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers,
      /*PJRT_CrossHostTransfers_PJRT_Buffer_CopyToRemoteDevice=*/
      PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice};
}

}  // namespace pjrt
