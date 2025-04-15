/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/dcn_utils.h"

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::MicroToNano;
using tsl::profiler::StatType;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XStatVisitor;

DcnMessage CreateDcnMessageFromStats(const XEventVisitor& event_visitor) {
  DcnMessage dcn_message;
  event_visitor.ForEachStat([&](const XStatVisitor& stat) {
    if (!stat.Type()) return;
    switch (static_cast<StatType>(*stat.Type())) {
      case StatType::kDcnLabel: {
        dcn_message.collective_name = stat.ToString();
        break;
      }
      case StatType::kDcnSourceSliceId: {
        dcn_message.slice_src = stat.IntValue();
        break;
      }
      case StatType::kDcnSourcePerSliceDeviceId: {
        dcn_message.tpu_src = stat.IntValue();
        break;
      }
      case StatType::kDcnDestinationSliceId: {
        dcn_message.slice_dst = stat.IntValue();
        break;
      }
      case StatType::kDcnDestinationPerSliceDeviceId: {
        dcn_message.tpu_dst = stat.IntValue();
        break;
      }
      case StatType::kDcnChunk: {
        dcn_message.chunk_id = stat.IntValue();
        break;
      }
      case StatType::kDcnLoopIndex: {
        dcn_message.loop_index_id = stat.IntValue();

        break;
      }
      case StatType::kPayloadSizeBytes: {
        dcn_message.size_bytes = stat.IntValue();
        break;
      }
      case StatType::kDuration: {
        dcn_message.duration_us = stat.IntOrUintValue();
        dcn_message.start_timestamp_ns =
            event_visitor.TimestampNs() - MicroToNano(dcn_message.duration_us);
        dcn_message.end_timestamp_ns = event_visitor.TimestampNs();
        break;
      }
      default:
        break;
    }
  });
  return dcn_message;
}

// Analyze message to see if it can be directly processed or it falls under
// corner-case categories, or if there is something wrong with it.
void SetMessageValidity(DcnMessage& dcn_message) {
  // Message should not be valid if fields have not been set properly
  // The main use of that is to detect unexpected key format changes that do
  // not cause crashes.
  if (dcn_message.collective_name.empty() || dcn_message.slice_src == -1 ||
      dcn_message.tpu_src == -1 || dcn_message.slice_dst == -1 ||
      dcn_message.tpu_dst == -1 || dcn_message.size_bytes == -1) {
    dcn_message.validity_info = DCN_MESSAGE_INVALID_BAD_KEY;
  } else if (dcn_message.duration_us == 0) {
    // Destination timestamp smaller than the source timestamp likely due to
    // clock skew
    dcn_message.validity_info = DCN_MESSAGE_INVALID_CLOCK_SKEW;
  } else if (dcn_message.slice_src == dcn_message.slice_dst) {
    // Loopback messages remain on the same host, so they are valid
    // even though they should not go through DCN.
    // TODO(emizan): Get host/TPU info and check host, not slice.
    dcn_message.validity_info = DCN_MESSAGE_VALID_LOOPBACK;
  } else {
    dcn_message.validity_info = DCN_MESSAGE_VALID;
  }
}
}  // namespace

DcnMessage GetDcnMessageFromXEvent(const XEventVisitor& event_visitor) {
  DcnMessage dcn_message = CreateDcnMessageFromStats(event_visitor);
  SetMessageValidity(dcn_message);
  return dcn_message;
}

bool IsDcnEvent(const tsl::profiler::XEventVisitor& event) {
  return absl::StartsWith(event.Name(), "MegaScale:");
}

}  // namespace profiler
}  // namespace tensorflow
