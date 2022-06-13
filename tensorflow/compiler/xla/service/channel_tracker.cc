/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/channel_tracker.h"

#include <memory>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

ChannelTracker::ChannelTracker() : next_channel_(1) {}

StatusOr<ChannelHandle> ChannelTracker::NewChannel(
    ChannelHandle::ChannelType type) {
  if (type != ChannelHandle::DEVICE_TO_DEVICE &&
      type != ChannelHandle::HOST_TO_DEVICE &&
      type != ChannelHandle::DEVICE_TO_HOST) {
    return InvalidArgument("Invalid channel type: %d", type);
  }
  absl::MutexLock lock(&channel_mutex_);

  // Create a new channel handle with a unique value.
  ChannelHandle new_handle = AllocateHandle(type);

  // Register a channel object associated with the handle.
  Channel channel;
  channel.has_sender = false;
  channel.receiver_count = 0;
  channel.type = type;
  opaque_to_channel_[new_handle.handle()] = channel;

  return new_handle;
}

Status ChannelTracker::RegisterSend(const ChannelHandle& handle) {
  absl::MutexLock lock(&channel_mutex_);
  return RegisterSendInternal(handle);
}

Status ChannelTracker::RegisterRecv(const ChannelHandle& handle) {
  absl::MutexLock lock(&channel_mutex_);
  return RegisterRecvInternal(handle);
}

ChannelHandle ChannelTracker::AllocateHandle(ChannelHandle::ChannelType type) {
  int64_t handle_value = next_channel_++;
  ChannelHandle result;
  result.set_handle(handle_value);
  result.set_type(type);
  return result;
}

Status ChannelTracker::RegisterSendInternal(const ChannelHandle& handle) {
  if (!opaque_to_channel_.contains(handle.handle())) {
    return NotFound("channel handle not found: %d", handle.handle());
  }
  Channel& channel = opaque_to_channel_[handle.handle()];
  if (channel.type == ChannelHandle::HOST_TO_DEVICE) {
    return FailedPrecondition(
        "host-to-device channels cannot be used with a Send operation; "
        "channel handle: %d",
        handle.handle());
  }

  if (channel.has_sender) {
    return FailedPrecondition(
        "when registering send, passed a channel handle that is already used "
        "by a sender: %d",
        handle.handle());
  }
  channel.has_sender = true;
  return OkStatus();
}

Status ChannelTracker::RegisterRecvInternal(const ChannelHandle& handle) {
  if (!opaque_to_channel_.contains(handle.handle())) {
    return NotFound("channel handle not found: %d", handle.handle());
  }
  Channel& channel = opaque_to_channel_[handle.handle()];
  if (channel.type == ChannelHandle::DEVICE_TO_HOST) {
    return FailedPrecondition(
        "device-to-host channels cannot be used with a Recv operation; "
        "channel handle: %d",
        handle.handle());
  }

  // TODO(b/33942691): Allow more than 1 receivers for broadcast.
  if (channel.receiver_count >= 1) {
    return FailedPrecondition(
        "when registering recv, passed a channel handle that is already used "
        "by a receiver: %d",
        handle.handle());
  }
  channel.receiver_count += 1;
  return OkStatus();
}

}  // namespace xla
