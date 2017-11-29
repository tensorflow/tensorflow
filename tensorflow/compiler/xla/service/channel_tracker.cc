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

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

ChannelTracker::ChannelTracker() : next_channel_(1) {}

ChannelHandle ChannelTracker::NewChannel() {
  tensorflow::mutex_lock lock(channel_mutex_);

  // Create a new channel handle with a unique value.
  const ChannelHandle new_handle = AllocateHandle();

  // Register a channel object associated with the handle.
  Channel channel;
  channel.has_sender = false;
  channel.receiver_count = 0;
  opaque_to_channel_[new_handle.handle()] = channel;

  return new_handle;
}

Status ChannelTracker::RegisterSend(const ChannelHandle& handle) {
  tensorflow::mutex_lock lock(channel_mutex_);
  return RegisterSendInternal(handle);
}

Status ChannelTracker::RegisterRecv(const ChannelHandle& handle) {
  tensorflow::mutex_lock lock(channel_mutex_);
  return RegisterRecvInternal(handle);
}

ChannelHandle ChannelTracker::AllocateHandle() {
  int64 handle_value = next_channel_++;
  ChannelHandle result;
  result.set_handle(handle_value);
  return result;
}

Status ChannelTracker::RegisterSendInternal(const ChannelHandle& handle) {
  if (opaque_to_channel_.count(handle.handle()) == 0) {
    return NotFound("channel handle not found: %lld", handle.handle());
  }
  Channel& channel = opaque_to_channel_[handle.handle()];
  if (channel.has_sender) {
    return FailedPrecondition("channel handle is already used by a sender");
  }
  channel.has_sender = true;
  return Status::OK();
}

Status ChannelTracker::RegisterRecvInternal(const ChannelHandle& handle) {
  if (opaque_to_channel_.count(handle.handle()) == 0) {
    return NotFound("channel handle not found: %lld", handle.handle());
  }
  Channel& channel = opaque_to_channel_[handle.handle()];
  // TODO(b/33942691): Allow more than 1 receivers for broadcast.
  if (channel.receiver_count >= 1) {
    return FailedPrecondition("channel handle is already used by a receiver");
  }
  channel.receiver_count += 1;
  return Status::OK();
}

}  // namespace xla
