/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_trace_container.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tensorflow/core/util/proto/proto_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(XPlaneToTraceContainerTest, CounterLine) {
  XSpace xspace;
  CHECK_OK(tensorflow::proto_utils::ParseTextFormatFromString(
      absl::Substitute(
          "planes {"
          "  name: \"/device:GPU:0\""
          "  lines {"
          "    name: \"_counters_\""
          "    events {"
          "      metadata_id: 100"
          "      offset_ps: $0"
          "      stats { metadata_id: 200 uint64_value: 100 }"
          "    }"
          "    events {"
          "      metadata_id: 100"
          "      offset_ps: $1"
          "      stats { metadata_id: 200 uint64_value: 200 }"
          "    }"
          "    events {"
          "      metadata_id: 101"
          "      offset_ps: $0"
          "      stats { metadata_id: 201 uint64_value: 300 }"
          "    }"
          "    events {"
          "      metadata_id: 101"
          "      offset_ps: $1"
          "      stats { metadata_id: 201 uint64_value: 400 }"
          "    }"
          "  }"
          "  lines {"
          "    id: 14"
          "    name: \"Stream #14(MemcpyH2D)\""
          "    timestamp_ns: $3"
          "    events {"
          "      metadata_id: 10"
          "      offset_ps: 0"
          "      duration_ps: $1"
          "      stats { metadata_id: 8 uint64_value: 100 }"
          "      stats { metadata_id: 9 str_value: \"$$1\" }"
          "    }"
          "    events {"
          "      metadata_id: 10"
          "      offset_ps: $0"
          "      duration_ps: $3"
          "      stats { metadata_id: 8 uint64_value: 200 }"
          "      stats { metadata_id: 9 str_value: \"abcd\" }"
          "    }"
          "  }"
          "  event_metadata {key: 10 value: { id: 10 name: \"MemcpyD2D\" }}"
          "  event_metadata {key: 100 value: { id: 100 name: \"Counter 1\" }}"
          "  event_metadata {key: 101 value: { id: 101 name: \"Counter 2\" }}"
          "  stat_metadata {key: 8 value: { id: 8 name: \"RemoteCall\"}}"
          "  stat_metadata {key: 9 value: { id: 8 name: \"context_id\"}}"
          "  stat_metadata {key: 200 value: { id: 200 name: \"counter_1\"}}"
          "  stat_metadata {key: 201 value: { id: 201 name: \"counter_2\"}}"
          "}",
          tsl::profiler::UniToPico(1), tsl::profiler::UniToPico(2),
          tsl::profiler::UniToNano(1), tsl::profiler::UniToNano(500)),
      &xspace));
  TraceEventsContainer container;
  ConvertXSpaceToTraceEventsContainer("localhost", xspace, &container);
  absl::flat_hash_map<std::string, absl::flat_hash_map<uint64_t, uint64_t>>
      counter_offset_to_values;
  container.ForAllEvents([&counter_offset_to_values](const TraceEvent& event) {
    if (absl::StrContains(event.name(), "Counter")) {
      uint64_t offset = event.timestamp_ps();
      RawData raw_data;
      raw_data.ParseFromString(event.raw_data());
      counter_offset_to_values[event.name()][offset] =
          raw_data.args().arg(0).uint_value();
    }
  });
  EXPECT_THAT(
      counter_offset_to_values,
      UnorderedElementsAre(
          Pair("Counter 1",
               UnorderedElementsAre(Pair(tsl::profiler::UniToPico(1), 100),
                                    Pair(tsl::profiler::UniToPico(2), 200))),
          Pair("Counter 2",
               UnorderedElementsAre(Pair(tsl::profiler::UniToPico(1), 300),
                                    Pair(tsl::profiler::UniToPico(2), 400)))));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
