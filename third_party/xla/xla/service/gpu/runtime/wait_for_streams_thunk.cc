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

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "tsl/platform/errors.h"

namespace xla::gpu {

absl::Status WaitForStreamsThunk::ExecuteOnStream(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream,
                      Thunk::GetStreamForExecution(stream_id_, params));

  VLOG(5) << "Waiting for stream ids: "
          << absl::StrJoin(
                 wait_for_stream_ids_, ", ",
                 [&](std::string* s, const ExecutionStreamId& stream_id) {
                   absl::StrAppend(s, stream_id.value());
                 });
  for (const auto& stream_id : wait_for_stream_ids_) {
    TF_ASSIGN_OR_RETURN(se::Stream * wait_on_stream,
                        Thunk::GetStreamForExecution(stream_id, params));

    TF_RETURN_IF_ERROR(stream->WaitFor(wait_on_stream));
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
