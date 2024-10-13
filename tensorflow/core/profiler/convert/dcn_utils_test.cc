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

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::kMegaScaleDcnReceive;
using tsl::profiler::XEventBuilder;
using tsl::profiler::XEventVisitor;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XPlaneVisitor;

void PopulateXPlane(XPlane &xplane, absl::string_view event_name, int offset,
                    absl::string_view label, int64_t source_slice_id,
                    int64_t source_per_slice_device_id,
                    int64_t destination_slice_id,
                    int64_t destination_per_slice_device_id, int64_t chunk,
                    int64_t loop_index, int64_t payload_size,
                    int64_t duration) {
  XPlaneBuilder xplane_builder(&xplane);

  XEventMetadata *event_metadata = xplane_builder.GetOrCreateEventMetadata(1);
  event_metadata->set_name(std::string(event_name));

  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventBuilder event_builder = xline_builder.AddEvent(*event_metadata);
  event_builder.SetOffsetNs(offset);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_label"), label);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_slice_id"),
      source_slice_id);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_per_slice_device_id"),
      source_per_slice_device_id);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_destination_slice_id"),
      destination_slice_id);
  event_builder.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                                 "dcn_destination_per_slice_device_id"),
                             destination_per_slice_device_id);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_chunk"), chunk);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_loop_index"), loop_index);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("duration_us"), duration);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("payload_size_bytes"),
      payload_size);
}

TEST(DcnUtilsTest, IsDcnEvent) {
  XPlane xplane;
  PopulateXPlane(xplane, kMegaScaleDcnReceive, 0, "test", 0, 0, 0, 0, 0, 0, 0,
                 0);
  XLine line = xplane.lines()[0];
  XPlaneVisitor xplane_visitor = tsl::profiler::CreateTfXPlaneVisitor(&xplane);

  XEventVisitor visitor(&xplane_visitor, &line, &line.events()[0]);
  EXPECT_TRUE(IsDcnEvent(visitor));
}

TEST(DcnUtilsTest, IsNotDcnEvent) {
  XPlane xplane;
  PopulateXPlane(xplane, "test", 0, "test", 0, 0, 0, 0, 0, 0, 0, 0);
  XLine line = xplane.lines()[0];
  XPlaneVisitor xplane_visitor = tsl::profiler::CreateTfXPlaneVisitor(&xplane);

  XEventVisitor visitor(&xplane_visitor, &line, &line.events()[0]);
  EXPECT_FALSE(IsDcnEvent(visitor));
}

TEST(DcnUtilsTest, GetDcnMessageFromXEvent) {
  XPlane xplane;
  PopulateXPlane(xplane, kMegaScaleDcnReceive, 100000, "all-reduce.273_312", 2,
                 3, 1, 3, 0, 24, 32768, 50);
  XPlaneVisitor xplane_visitor = tsl::profiler::CreateTfXPlaneVisitor(&xplane);
  XEventVisitor visitor(&xplane_visitor, &xplane.lines()[0],
                        &xplane.lines()[0].events()[0]);
  EXPECT_THAT(GetDcnMessageFromXEvent(visitor),
              testing::FieldsAre(
                  "all-reduce.273_312", /* collective name */
                  2, 3, 1, 3, /* slice_src, tpu_src, slice_dst, tpu_dst */
                  /* start_timestamp_ns, end_timestamp_ns, duration_us */
                  50000, 100000, 50,
                  /* size_bytes, chunk_id, loop_index_id */
                  32768, 0, 24,
                  /* validity_info */
                  DCN_MESSAGE_VALID));
}

TEST(DcnUtilsTest, GetDcnMessageFromXEventLoopBack) {
  XPlane xplane;
  PopulateXPlane(xplane, kMegaScaleDcnReceive, 5000000, "all-gather.1234", 2, 3,
                 2, 1, 4, 40, 1000, 1000);
  XPlaneVisitor xplane_visitor = tsl::profiler::CreateTfXPlaneVisitor(&xplane);
  XEventVisitor visitor(&xplane_visitor, &xplane.lines()[0],
                        &xplane.lines()[0].events()[0]);
  EXPECT_THAT(GetDcnMessageFromXEvent(visitor),
              testing::FieldsAre(
                  "all-gather.1234", /* collective name */
                  2, 3, 2, 1, /* slice_src, tpu_src, slice_dst, tpu_dst */
                  /* start_timestamp_ns. end_timestamp_ns, duration_us */
                  4000000, 5000000, 1000,
                  /* size_bytes, chunk_id, loop_index_id */
                  1000, 4, 40,
                  /* validity_info */
                  DCN_MESSAGE_VALID_LOOPBACK));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
