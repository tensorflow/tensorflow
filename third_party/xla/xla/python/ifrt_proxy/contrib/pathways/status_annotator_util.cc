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

#include "xla/python/ifrt_proxy/contrib/pathways/status_annotator_util.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_registry.h"
#include "xla/python/ifrt_proxy/contrib/pathways/status_annotator.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_to_from_proto.h"

namespace ifrt_proxy_contrib_pathways {

static constexpr absl::string_view kObjectStoreDumpPayloadUrl =
    "type.googleapis.com/ifrt_proxy_contrib_pathways.ObjectStoreDumpProto";

static void ExpandObjectStoreDump(absl::Status& status) {
  std::optional<absl::Cord> payload =
      status.GetPayload(kObjectStoreDumpPayloadUrl);
  if (!payload.has_value()) {
    return;
  }
  ObjectStoreDumpProto object_store_dump;
  if (!object_store_dump.ParseFromString(payload->Flatten())) {
    LOG(WARNING) << "Unable to expand string to ObjectStoreDumpProto: "
                 << payload->Flatten();
    tsl::errors::AppendToMessage(
        &status,
        "\nWARNING: Unable to parse attached payload string to "
        "ObjectStoreDumpProto. Please see logs for the actual payload string.");
    return;
  }
  status.ErasePayload(kObjectStoreDumpPayloadUrl);

  std::string header = absl::StrCat(
      "Pathways object-store summary for device ", object_store_dump.device(),
      " at ", absl::FromUnixNanos(object_store_dump.dump_timestamp_ns()));
  absl::Status error = tsl::StatusFromProto(object_store_dump.dump_failed());
  if (!error.ok()) {
    tsl::errors::AppendToMessage(
        &status, "\n", header, " got error while dumping: ", error.ToString());
    return;
  }

  std::vector<std::string> cited_traces;
  tsl::errors::AppendToMessage(&status, "\n", header, ":");
  for (const auto& per_error_context : object_store_dump.per_error_context()) {
    xla::ifrt::TrackedUserContextRef tracked_user_context =
        xla::ifrt::UserContextRegistry::Get().Lookup(
            xla::ifrt::UserContextId(per_error_context.error_context_id()));
    if (tracked_user_context != nullptr) {
      std::string trace = tracked_user_context->user_context()->DebugString();
      absl::StrReplaceAll({{"\n", "\t"}}, &trace);
      absl::StrReplaceAll({{"\t", "\n                "}}, &trace);
      cited_traces.push_back(trace);
      tsl::errors::AppendToMessage(
          &status, "  - The following entries arise from user stack [",
          cited_traces.size(), "]:");
    } else {
      tsl::errors::AppendToMessage(
          &status,
          "  - The following entries arise from an unknown user stack:");
    }
    for (const auto& per_creator : per_error_context.per_creator()) {
      tsl::errors::AppendToMessage(&status, "      + ", per_creator.creator(),
                                   " with ", per_creator.ready_obj_count(),
                                   " 'ready' buffers of total size ",
                                   per_creator.ready_total_size(), " and ",
                                   per_creator.not_ready_obj_count(),
                                   " 'not ready' buffers of total size ",
                                   per_creator.not_ready_total_size());
    }
  }
  for (int i = 0; i < cited_traces.size(); ++i) {
    tsl::errors::AppendToMessage(
        &status, absl::StrFormat("[%3d]   %s", i + 1, cited_traces[i]));
  }
}

void AnnotateIfrtUserStatusWithObjectStoreDump(
    absl::Status& status, const ObjectStoreDumpProto& object_store_dump) {
  status.SetPayload(kObjectStoreDumpPayloadUrl,
                    object_store_dump.SerializeAsCord());
}

static const bool register_expanders = []() {
  xla::ifrt::CustomStatusExpanderRegistry::Get().Register(
      kObjectStoreDumpPayloadUrl, ExpandObjectStoreDump);
  return true;
}();

}  // namespace ifrt_proxy_contrib_pathways
