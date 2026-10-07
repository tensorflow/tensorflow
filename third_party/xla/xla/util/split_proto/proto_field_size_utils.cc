/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/util/split_proto/proto_field_size_utils.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/wire_format.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

constexpr absl::string_view kSeparator =
    "========================================"
    "========================================\n";
constexpr size_t kOneMib = 1024 * 1024;
constexpr double kDrillDownThresholdFraction = 0.01;
constexpr int kMaxDrillDownDepth = 3;
constexpr int kMaxSubfieldsPerField = 5;

struct SubfieldEntry {
  std::string name;
  const tsl::protobuf::FieldDescriptor* field = nullptr;
  size_t byte_size = 0;
};

struct RootFieldInfo {
  const tsl::protobuf::FieldDescriptor* descriptor = nullptr;
  size_t byte_size = 0;
  std::vector<SubfieldEntry> subfields;
};

// Iterates over all (sub-)messages within `field` of `msg` and invokes `fn` on
// each one.  Handles both singular and repeated message fields.
template <typename Fn>
void ForEachSubMessage(const tsl::protobuf::Message& msg,
                       const tsl::protobuf::Reflection* reflection,
                       const tsl::protobuf::FieldDescriptor* field, Fn fn) {
  if (!field->is_repeated()) {
    fn(reflection->GetMessage(msg, field));
  } else {
    int field_count = reflection->FieldSize(msg, field);
    for (int i = 0; i < field_count; ++i) {
      fn(reflection->GetRepeatedMessage(msg, field, i));
    }
  }
}

void CollectSubfields(const tsl::protobuf::Message& msg,
                      absl::string_view prefix, int depth, size_t threshold,
                      std::vector<SubfieldEntry>& subfields) {
  const tsl::protobuf::Reflection* reflection = msg.GetReflection();
  if (reflection == nullptr) {
    return;
  }

  std::vector<const tsl::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(msg, &fields);

  for (const tsl::protobuf::FieldDescriptor* field : fields) {
    size_t field_size =
        tsl::protobuf::internal::WireFormat::FieldByteSize(field, msg);
    if (field_size < threshold) {
      continue;
    }

    std::string full_name;
    if (prefix.empty()) {
      full_name = field->name();
    } else {
      full_name = absl::StrCat(prefix, ".", field->name());
    }

    if (field->cpp_type() == tsl::protobuf::FieldDescriptor::CPPTYPE_MESSAGE &&
        depth < kMaxDrillDownDepth) {
      size_t prev_size = subfields.size();
      ForEachSubMessage(msg, reflection, field,
                        [&](const tsl::protobuf::Message& child_msg) {
                          CollectSubfields(child_msg, full_name, depth + 1,
                                           threshold, subfields);
                        });
      // If recursion produced more granular child entries, skip adding the
      // coarser parent message to avoid double-counting and wasting display
      // slots on redundant intermediate wrappers.
      //
      // Without this continue:
      //   1. root_field (tag 1, type MESSAGE): 50.10 MiB
      //      -> container (tag 2, type MESSAGE): 50.05 MiB       <-- redundant!
      //      -> container.payload (tag 1, type STRING): 48.00 MiB <-- culprit
      //      -> container.metadata (tag 2, type BYTES): 2.00 MiB
      //
      // With this continue:
      //   1. root_field (tag 1, type MESSAGE): 50.10 MiB
      //      -> container.payload (tag 1, type STRING): 48.00 MiB
      //      -> container.metadata (tag 2, type BYTES): 2.00 MiB
      if (subfields.size() > prev_size) {
        continue;
      }
    }

    auto it = absl::c_find_if(
        subfields, [&](const SubfieldEntry& e) { return e.name == full_name; });
    if (it != subfields.end()) {
      it->byte_size += field_size;
    } else {
      SubfieldEntry entry;
      entry.name = full_name;
      entry.field = field;
      entry.byte_size = field_size;
      subfields.push_back(std::move(entry));
    }
  }
}

std::string FormatMib(size_t bytes) {
  return absl::StrFormat("%.2f MiB", static_cast<double>(bytes) / kOneMib);
}

}  // namespace

std::string GetTopKProtoFieldSizes(const tsl::protobuf::MessageLite& message,
                                   int top_k) {
  const auto* full_msg =
      tsl::protobuf::DynamicCastMessage<tsl::protobuf::Message>(&message);
  if (full_msg == nullptr) {
    const size_t total_bytes = message.ByteSizeLong();
    return absl::StrCat(
        kSeparator,
        absl::StrFormat("Proto [%s] (reflection not supported)\n",
                        message.GetTypeName()),
        absl::StrFormat("Total ByteSize: %s (%zu bytes)\n",
                        FormatMib(total_bytes), total_bytes),
        kSeparator);
  }

  size_t total_byte_size = full_msg->ByteSizeLong();
  const tsl::protobuf::Reflection* reflection = full_msg->GetReflection();
  if (reflection == nullptr) {
    return absl::StrCat(
        kSeparator,
        absl::StrFormat("Proto [%s] (reflection unavailable)\n",
                        message.GetTypeName()),
        absl::StrFormat("Total ByteSize: %s (%zu bytes)\n",
                        FormatMib(total_byte_size), total_byte_size),
        kSeparator);
  }

  std::vector<const tsl::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(*full_msg, &fields);

  std::vector<RootFieldInfo> root_fields;
  root_fields.reserve(fields.size());

  size_t drill_down_threshold = std::max(
      static_cast<size_t>(kDrillDownThresholdFraction * total_byte_size),
      kOneMib);

  for (const tsl::protobuf::FieldDescriptor* field : fields) {
    size_t field_size =
        tsl::protobuf::internal::WireFormat::FieldByteSize(field, *full_msg);
    RootFieldInfo info;
    info.descriptor = field;
    info.byte_size = field_size;

    if (field->cpp_type() == tsl::protobuf::FieldDescriptor::CPPTYPE_MESSAGE &&
        field_size >= drill_down_threshold) {
      ForEachSubMessage(*full_msg, reflection, field,
                        [&](const tsl::protobuf::Message& sub_msg) {
                          CollectSubfields(sub_msg, /*prefix=*/"", /*depth=*/1,
                                           drill_down_threshold,
                                           info.subfields);
                        });
      absl::c_stable_sort(info.subfields,
                          [](const SubfieldEntry& a, const SubfieldEntry& b) {
                            if (a.byte_size != b.byte_size) {
                              return a.byte_size > b.byte_size;
                            }
                            return a.field->number() < b.field->number();
                          });
    }

    root_fields.push_back(std::move(info));
  }

  int num_to_display =
      std::min(static_cast<int>(root_fields.size()), std::max(0, top_k));

  absl::c_partial_sort(root_fields, root_fields.begin() + num_to_display,
                       [](const RootFieldInfo& a, const RootFieldInfo& b) {
                         if (a.byte_size != b.byte_size) {
                           return a.byte_size > b.byte_size;
                         }
                         return a.descriptor->number() < b.descriptor->number();
                       });

  std::string output;
  absl::StrAppend(&output, kSeparator);
  absl::StrAppendFormat(&output, "Top %d largest fields in proto [%s]\n",
                        num_to_display, full_msg->GetTypeName());
  absl::StrAppendFormat(&output, "Total ByteSize: %s (%zu bytes)\n",
                        FormatMib(total_byte_size), total_byte_size);
  absl::StrAppend(&output, kSeparator);

  for (int i = 0; i < num_to_display; ++i) {
    const auto& root_field = root_fields[i];
    double pct = total_byte_size > 0
                     ? (static_cast<double>(root_field.byte_size) /
                        static_cast<double>(total_byte_size)) *
                           100.0
                     : 0.0;
    absl::StrAppendFormat(
        &output, "  %d. %s (tag %d, type %s): %s (%.2f%%)\n", i + 1,
        root_field.descriptor->name(), root_field.descriptor->number(),
        absl::AsciiStrToUpper(root_field.descriptor->type_name()),
        FormatMib(root_field.byte_size), pct);

    int sub_count = 0;
    for (const auto& sub : root_field.subfields) {
      absl::StrAppendFormat(&output, "     -> %s (tag %d, type %s): %s\n",
                            sub.name, sub.field->number(),
                            absl::AsciiStrToUpper(sub.field->type_name()),
                            FormatMib(sub.byte_size));
      if (++sub_count >= kMaxSubfieldsPerField) {
        break;
      }
    }
  }

  absl::StrAppend(&output, kSeparator);
  return output;
}

absl::Status AnnotateResourceExhaustedError(
    const absl::Status& status, const tsl::protobuf::MessageLite& record,
    int top_k) {
  if (!absl::IsResourceExhausted(status)) {
    return status;
  }
  std::string annotation = GetTopKProtoFieldSizes(record, top_k);
  absl::Status new_status(
      status.code(), status.message().empty()
                         ? annotation
                         : absl::StrCat(status.message(), "\n\n", annotation));
  status.ForEachPayload(
      [&](absl::string_view type_url, const absl::Cord& payload) {
        new_status.SetPayload(type_url, payload);
      });
  return new_status;
}

}  // namespace xla
