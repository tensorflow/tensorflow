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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
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
#include "tsl/platform/fingerprint.h"

namespace ifrt_proxy_contrib_pathways {

static constexpr absl::string_view kObjectStoreDumpPayloadUrl =
    "type.googleapis.com/ifrt_proxy_contrib_pathways.ObjectStoreDumpProto";

static constexpr int kMaxCitedTraces = 20;

struct PerStackTrace {
  struct PerCreator {
    int64_t ready_obj_count = 0;
    int64_t ready_total_size = 0;
    int64_t not_ready_obj_count = 0;
    int64_t not_ready_total_size = 0;
  };
  std::string stack_trace;
  absl::flat_hash_map<std::string, PerCreator> per_creator;
  int64_t total_size = 0;
};

std::vector<std::unique_ptr<PerStackTrace>> SortedPerStackTrace(
    const ObjectStoreDumpProto& object_store_dump) {
  std::vector<std::unique_ptr<PerStackTrace>> result;
  absl::flat_hash_map<uint64_t, PerStackTrace*> per_stack_trace;
  for (const auto& per_error_context :
       object_store_dump.dump().per_error_context()) {
    xla::ifrt::TrackedUserContextRef tracked_user_context =
        xla::ifrt::UserContextRegistry::Get().Lookup(
            xla::ifrt::UserContextId(per_error_context.error_context_id()));
    std::string stack_trace = "unknown";
    if (tracked_user_context != nullptr) {
      stack_trace = tracked_user_context->user_context()->DebugString();
    }
    uint64_t stack_trace_fprint = tsl::Fingerprint64(stack_trace);

    PerStackTrace* entry = nullptr;
    if (auto it = per_stack_trace.find(stack_trace_fprint);
        it != per_stack_trace.end()) {
      entry = it->second;
    } else {
      entry = result.emplace_back(std::make_unique<PerStackTrace>()).get();
      entry->stack_trace = stack_trace;
      per_stack_trace.insert(it, {stack_trace_fprint, entry});
    }

    for (const auto& per_creator : per_error_context.per_creator()) {
      PerStackTrace::PerCreator& creator =
          entry->per_creator[per_creator.creator()];
      creator.ready_obj_count += per_creator.ready_obj_count();
      creator.ready_total_size += per_creator.ready_total_size();
      creator.not_ready_obj_count += per_creator.not_ready_obj_count();
      creator.not_ready_total_size += per_creator.not_ready_total_size();
      entry->total_size +=
          per_creator.ready_total_size() + per_creator.not_ready_total_size();
    }
  }

  auto is_bigger = [](const std::unique_ptr<PerStackTrace>& a,
                      const std::unique_ptr<PerStackTrace>& b) {
    return (a->total_size > b->total_size);
  };

  std::sort(result.begin(), result.end(), is_bigger);

  return result;
}

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

  if (object_store_dump.has_dump_failed()) {
    absl::Status error = tsl::StatusFromProto(object_store_dump.dump_failed());
    tsl::errors::AppendToMessage(
        &status, "\n", header, " got error while dumping: ", error.ToString());
    return;
  }

  auto sorted_per_stack_trace = SortedPerStackTrace(object_store_dump);
  absl::StrAppend(&header, " (showing ",
                  std::min(kMaxCitedTraces,
                           static_cast<int>(sorted_per_stack_trace.size())),
                  " of ", sorted_per_stack_trace.size(), " entries)");

  if (!object_store_dump.dump().warning().empty()) {
    absl::StrAppend(&header, " (warning: ", object_store_dump.dump().warning(),
                    ")");
  }

  std::vector<std::string> cited_traces;
  tsl::errors::AppendToMessage(&status, "\n", header, ":");
  for (const auto& per_error_context : sorted_per_stack_trace) {
    std::string stack_trace = std::move(per_error_context->stack_trace);
    if (stack_trace != "unknown") {
      absl::StrReplaceAll({{"\n", "\t"}}, &stack_trace);
      absl::StrReplaceAll({{"\t", "\n                "}}, &stack_trace);
      cited_traces.push_back(stack_trace);
      tsl::errors::AppendToMessage(
          &status, "  - The following entries arise from user stack [",
          cited_traces.size(), "]:");
    } else {
      tsl::errors::AppendToMessage(
          &status,
          "  - The following entries arise from an unknown user stack:");
    }
    for (const auto& [creator, per_creator] : per_error_context->per_creator) {
      tsl::errors::AppendToMessage(
          &status, "      + ", creator, " with ", per_creator.ready_obj_count,
          " 'ready' buffers of total size ", per_creator.ready_total_size,
          " and ", per_creator.not_ready_obj_count,
          " 'not ready' buffers of total size ",
          per_creator.not_ready_total_size);
    }
    if (cited_traces.size() >= kMaxCitedTraces) {
      break;
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
