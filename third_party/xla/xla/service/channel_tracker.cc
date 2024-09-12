/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/channel_tracker.h"

#include "xla/util.h"

namespace xla {

absl::StatusOr<ChannelHandle> ChannelTracker::NewChannel(
    ChannelHandle::ChannelType type) {
  if (type != ChannelHandle::DEVICE_TO_DEVICE &&
      type != ChannelHandle::HOST_TO_DEVICE &&
      type != ChannelHandle::DEVICE_TO_HOST) {
    return InvalidArgument("Invalid channel type: %d", type);
  }
  absl::MutexLock lock(&channel_mutex_);

  // Create a new channel handle with a unique value.
  ChannelHandle new_handle;
  new_handle.set_handle(next_channel_++);
  new_handle.set_type(type);

  return new_handle;
}

}  // namespace xla
