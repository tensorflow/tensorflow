/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/wait_for_streams_thunk.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

absl::Status WaitForStreamsThunk::ExecuteOnStream(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream,
                      Thunk::GetStreamForExecution(stream_id_, params));

  VLOG(5) << "Waiting for stream id: " << wait_for_stream_id_;
  TF_ASSIGN_OR_RETURN(
      se::Stream * wait_on_stream,
      Thunk::GetStreamForExecution(wait_for_stream_id_, params));

  return stream->WaitFor(wait_on_stream);
}

}  // namespace xla::gpu
