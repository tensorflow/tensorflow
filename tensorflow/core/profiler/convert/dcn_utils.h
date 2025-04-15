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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_DCN_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_DCN_UTILS_H_

#include <string>

#include "xla/tsl/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// DCN Message Validity
enum DcnMessageValidity {
  // Valid message
  DCN_MESSAGE_VALID = 1,
  // Valid message, but should not go through DCN, so it should not use BW.
  DCN_MESSAGE_VALID_LOOPBACK = 2,
  // Invalid message with 0 duration due to clock skew. Should be ignored.
  DCN_MESSAGE_INVALID_CLOCK_SKEW = 3,
  // Message that cannot be decoded. Should be ignored.
  DCN_MESSAGE_INVALID_BAD_KEY = 4
};

// Structure representing a DCN event
struct DcnMessage {
  // Unique collective that generated this message, format should be
  // <col name>_<number>, e.g. all_gather_34
  std::string collective_name = "";
  // Src info
  // TODO(emizan) Add host info when you figure out how to get it from
  // slice+tpu.
  int32_t slice_src = -1;
  int32_t tpu_src = -1;
  // Dst info
  int32_t slice_dst = -1;
  int32_t tpu_dst = -1;
  // Timing info in ns. Since MSXLA TraceMe's have us timestamps, we need to
  // multiply by 1000 to get these timestamps.
  uint64_t start_timestamp_ns = 0;
  uint64_t end_timestamp_ns = 0;
  uint64_t duration_us = 0;
  // Size info
  size_t size_bytes = 0;
  // Chunk and Loop index
  int32_t chunk_id = -1;
  int32_t loop_index_id = -1;
  // Is message valid/invalid and why
  DcnMessageValidity validity_info = DCN_MESSAGE_INVALID_BAD_KEY;
  // TBD: Add flow events in case you need to connect to other events pointed to
  // by MSXLA TraceMe's
};

DcnMessage GetDcnMessageFromXEvent(
    const tsl::profiler::XEventVisitor& event_visitor);

// Check if the XEventVisitor is a DCN Message
bool IsDcnEvent(const tsl::profiler::XEventVisitor& event);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_DCN_UTILS_H_
