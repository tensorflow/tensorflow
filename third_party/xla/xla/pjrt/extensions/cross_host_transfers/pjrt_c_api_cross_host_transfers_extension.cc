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
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/shape.h"

namespace pjrt {

PJRT_Error* PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers(
    PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args",
      PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args_STRUCT_SIZE,
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

  std::vector<xla::PjRtGlobalDeviceId> src_global_device_ids;
  src_global_device_ids.reserve(args->num_shapes);

  std::vector<xla::CrossHostTransferKey> transfer_keys;
  transfer_keys.reserve(args->num_shapes);

  for (int i = 0; i < args->num_shapes; ++i) {
    src_global_device_ids.push_back(args->src_global_device_ids[i]);
    transfer_keys.push_back(args->transfer_keys[i]);
  }

  PJRT_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers,
                        args->client->client->CrossHostReceiveBuffers(
                            args->device->device, shapes, src_global_device_ids,
                            std::move(transfer_keys)));

  for (int i = 0; i < buffers.size(); ++i) {
    args->buffers[i] = new PJRT_Buffer{std::move(buffers[i]), args->client};
  }
  return nullptr;
}

PJRT_Error* PJRT_Transfers_PJRT_Client_CrossHostSendBuffers(
    PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args* args) {
  std::vector<xla::PjRtBuffer*> buffers;
  buffers.reserve(args->num_buffers);
  for (int i = 0; i < args->num_buffers; ++i) {
    buffers.push_back(args->buffers[i]->buffer.get());
  }

  std::vector<xla::PjRtGlobalDeviceId> dst_global_device_ids;
  dst_global_device_ids.reserve(args->num_buffers);

  std::vector<xla::CrossHostTransferKey> transfer_keys;
  transfer_keys.reserve(args->num_buffers);

  for (int i = 0; i < args->num_buffers; ++i) {
    dst_global_device_ids.push_back(args->dst_global_device_ids[i]);
    transfer_keys.push_back(args->transfer_keys[i]);
  }

  PJRT_ASSIGN_OR_RETURN(
      std::vector<tsl::Future<>> send_futures,
      args->client->client->CrossHostSendBuffers(buffers, dst_global_device_ids,
                                                 std::move(transfer_keys)));

  for (int i = 0; i < buffers.size(); ++i) {
    args->send_events[i] = new PJRT_Event{std::move(send_futures[i])};
  }
  return nullptr;
}

namespace {
static xla::PjRtCrossHostRecvNotifier CCrossHostRecvNotifierToCpp(
    const PJRT_Transfers_CrossHostRecvNotifierInfo& c_notifier) {
  return [user_arg = c_notifier.user_arg, notifier = c_notifier.notifier](
             absl::StatusOr<xla::PjRtCrossHostRecvState> recv_state) {
    if (!recv_state.ok()) {
      auto error = new PJRT_Error{recv_state.status()};
      notifier(error, nullptr, nullptr, 0, user_arg);
      return;
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
    notifier(nullptr, serialized_descriptors.data(), descriptors_sizes.data(),
             descriptors.size(), user_arg);
  };
}
}  // namespace

PJRT_Transfers_CrossHostRecvNotifierInfo CppCrossHostRecvNotifierToC(
    const PJRT_Api* c_api, xla::PjRtCrossHostRecvNotifier cpp_notifier) {
  using CrossHostRecvNotifierFunction =
      std::function<void(PJRT_Error*, const char**, size_t*, size_t)>;
  auto notifier_function = new CrossHostRecvNotifierFunction(
      [cpp_notifier = std::move(cpp_notifier), c_api](
          PJRT_Error* error, const char** serialized_descriptors,
          size_t* descriptors_sizes, size_t num_descriptors) {
        if (error != nullptr) {
          absl::Status state = ::pjrt::PjrtErrorToStatus(error, c_api);
          return cpp_notifier(std::move(state));
        }
        xla::PjRtCrossHostRecvState state;
        state.descriptors.reserve(num_descriptors);
        for (int i = 0; i < num_descriptors; ++i) {
          xla::PjRtCrossHostRecvDescriptors descriptors;
          descriptors.serialized_descriptors.push_back(
              std::string(serialized_descriptors[i], descriptors_sizes[i]));
          state.descriptors.push_back(std::move(descriptors));
        }

        // TODO(emilyaf): Support cancellation.
        xla::PjRtCrossHostSendCancelNotifier cancel_notifier =
            [](absl::string_view, absl::Status,
               std::function<void(absl::Status)>) {
              LOG(FATAL) << "MakeCrossHostReceiveBuffers: Cancellation is not "
                            "supported in PJRT C API.";
            };
        state.cancel_notifier = cancel_notifier;
        return cpp_notifier(std::move(state));
      });
  return PJRT_Transfers_CrossHostRecvNotifierInfo{
      /*user_arg=*/notifier_function,
      /*notifier=*/
      [](PJRT_Error* error, const char** serialized_descriptors,
         size_t* descriptors_sizes, size_t num_descriptors, void* user_arg) {
        CrossHostRecvNotifierFunction* notifier_fn =
            reinterpret_cast<CrossHostRecvNotifierFunction*>(user_arg);
        (*notifier_fn)(error, serialized_descriptors, descriptors_sizes,
                       num_descriptors);
        delete notifier_fn;
      }};
}

PJRT_Transfers_CrossHostRemoteSendCallbackInfo
CppCrossHostRemoteSendCallbackToC(
    const PJRT_Api* c_api, xla::PjRtBuffer::RemoteSendCallback cpp_callback) {
  using RemoteSendCallbackFunction =
      std::function<void(PJRT_Error * error, bool sends_were_enqueued)>;
  auto on_done_function = new RemoteSendCallbackFunction(
      [cpp_callback = std::move(cpp_callback), c_api](
          PJRT_Error* error, bool sends_were_enqueued) {
        absl::Status status = ::pjrt::PjrtErrorToStatus(error, c_api);
        cpp_callback(status, sends_were_enqueued);
      });
  return PJRT_Transfers_CrossHostRemoteSendCallbackInfo{
      /*user_arg=*/on_done_function,
      /*on_done=*/
      [](PJRT_Error* error, bool sends_were_enqueued, void* user_arg) {
        RemoteSendCallbackFunction* on_done_fn =
            reinterpret_cast<RemoteSendCallbackFunction*>(user_arg);
        (*on_done_fn)(error, sends_were_enqueued);
        delete on_done_fn;
      }};
}

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
  xla::Future<std::string> descriptor_future(std::move(serialized_descriptor));

  xla::PjRtBuffer::RemoteSendCallback on_done =
      [user_arg = args->on_done.user_arg, on_done = args->on_done.on_done](
          absl::Status status, bool sends_were_enqueued) {
        auto error = new PJRT_Error{status};
        on_done(error, sends_were_enqueued, user_arg);
        delete error;
      };

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
      PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice,
      /*PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers=*/
      PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers,
      /*PJRT_Transfers_PJRT_Client_CrossHostSendBuffers=*/
      PJRT_Transfers_PJRT_Client_CrossHostSendBuffers,
  };
}

}  // namespace pjrt
