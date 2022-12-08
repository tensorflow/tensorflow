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

#include "tensorflow/compiler/xla/python/util.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#ifdef JAX_ENABLE_IFRT
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#endif
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

#ifdef JAX_ENABLE_IFRT
Status AwaitBuffersReady(ifrt::Array* ifrt_array) {
  Status s = ifrt_array->GetReadyFuture().Await();
  if (!s.ok()) {
    // Fix up error string because some clients rely on it.
    if (s.error_message() ==
        "GetReadyFuture() called on deleted or donated buffer") {
      s = InvalidArgument(
          "BlockHostUntilReady() called on deleted or donated buffer");
    }
  }
  return s;
}
#else
Status AwaitBuffersReady(
    absl::Span<const std::shared_ptr<PjRtBuffer> > buffers) {
  std::vector<PjRtFuture<Status> > futures;
  futures.reserve(buffers.size());
  Status status = OkStatus();
  for (const auto& buffer : buffers) {
    futures.push_back(buffer->GetReadyFuture());
  }
  for (PjRtFuture<Status>& future : futures) {
    Status s = future.Await();
    if (!s.ok()) {
      // Fix up error string because some clients rely on it.
      if (s.error_message() ==
          "GetReadyFuture() called on deleted or donated buffer") {
        status = InvalidArgument(
            "BlockHostUntilReady() called on deleted or donated buffer");
      } else {
        status = std::move(s);
      }
    }
  }
  return status;
}
#endif

}  // namespace xla
