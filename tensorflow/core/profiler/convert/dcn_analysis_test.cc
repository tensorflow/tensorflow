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
#include "tensorflow/core/profiler/convert/dcn_analysis.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/convert/dcn_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

using tensorflow::profiler::DCN_MESSAGE_INVALID_BAD_KEY;
using tensorflow::profiler::DCN_MESSAGE_INVALID_CLOCK_SKEW;
using tensorflow::profiler::DCN_MESSAGE_VALID;
using tensorflow::profiler::DCN_MESSAGE_VALID_LOOPBACK;
using tensorflow::profiler::XEventBuilder;
using tensorflow::profiler::XEventMetadata;
using tensorflow::profiler::XLineBuilder;
using tensorflow::profiler::XPlane;
using tensorflow::profiler::XPlaneBuilder;
using tensorflow::profiler::XPlaneVisitor;
using tensorflow::profiler::XSpace;
using ::testing::FieldsAre;
using tsl::profiler::kMegaScaleDcnReceive;
using tsl::profiler::kMegaScaleDcnSend;

TEST(DcnAnalysis, SetupMessageInfoTest) {
  XSpace space;
  XPlane *host_trace = space.add_planes();
  XPlaneBuilder host_trace_builder(host_trace);

  XEventMetadata *event_metadata_1 =
      host_trace_builder.GetOrCreateEventMetadata(1);
  event_metadata_1->set_name(std::string(kMegaScaleDcnReceive));
  XEventMetadata *event_metadata_2 =
      host_trace_builder.GetOrCreateEventMetadata(2);
  event_metadata_2->set_name(std::string(kMegaScaleDcnSend));

  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_trace);
  DcnEventsProcessor dcn_events_processor(/*num_tpu_tensor_cores*/ 4,
                                          /*is_megacore*/ false);
  dcn_events_processor.SetupMessageInfo(plane);
  ASSERT_FALSE(dcn_events_processor.HasDcnMessages(kMegaScaleDcnSend));
  ASSERT_TRUE(dcn_events_processor.HasDcnMessages(kMegaScaleDcnReceive));
  ASSERT_FALSE(dcn_events_processor.HasDcnMessages("Another Message"));
  ASSERT_EQ(dcn_events_processor.MegaScaleMessageId(kMegaScaleDcnReceive), 1);
  ASSERT_EQ(dcn_events_processor.MegaScaleMessageId(kMegaScaleDcnSend),
            std::nullopt);
}

// Test processing of valid messages and that all of them are received.
TEST(DcnAnalysis, CreateMessageTestValidMessages) {
  XSpace space;
  XPlane *host_trace = space.add_planes();
  XPlaneBuilder xplane_builder(host_trace);

  XEventMetadata *event_metadata_1 = xplane_builder.GetOrCreateEventMetadata(1);
  event_metadata_1->set_name(std::string(kMegaScaleDcnReceive));

  XLineBuilder xline_builder_0 = xplane_builder.GetOrCreateLine(0);
  XLineBuilder xline_builder_1 = xplane_builder.GetOrCreateLine(1);

  // 1st event
  XEventBuilder event_builder = xline_builder_0.AddEvent(*event_metadata_1);
  event_builder.SetOffsetNs(100000);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_label"),
      "all-reduce.273_312");
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_slice_id"), 2);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_per_slice_device_id"),
      3);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_destination_slice_id"), 1);
  event_builder.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                                 "dcn_destination_per_slice_device_id"),
                             3);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_chunk"), 0);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_loop_index"), 24);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("duration_us"), 50);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("payload_size_bytes"), 32768);

  // 2nd event, same line
  event_builder = xline_builder_0.AddEvent(*event_metadata_1);
  event_builder.SetOffsetNs(175000);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_label"),
      "super-collective.1234");
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_slice_id"), 112);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_per_slice_device_id"),
      1);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_destination_slice_id"), 34);
  event_builder.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                                 "dcn_destination_per_slice_device_id"),
                             2);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_chunk"), 4);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_loop_index"), 0);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("duration_us"), 50);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("payload_size_bytes"), 1);

  // 3rd event event, new line, no chunk/loop index
  event_builder = xline_builder_1.AddEvent(*event_metadata_1);
  event_builder.SetOffsetNs(150000);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_label"), "super-collective");
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_slice_id"), 9);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_per_slice_device_id"),
      3);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_destination_slice_id"), 0);
  event_builder.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                                 "dcn_destination_per_slice_device_id"),
                             0);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("duration_us"), 75);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("payload_size_bytes"), 10);

  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_trace);
  DcnEventsProcessor dcn_events_processor(4, false);
  dcn_events_processor.SetupMessageInfo(plane);
  dcn_events_processor.ProcessReceiveMessages(plane);

  ASSERT_EQ(dcn_events_processor.NumReceivedMessages(), 3);
  EXPECT_THAT(dcn_events_processor.GetMessage(0),
              FieldsAre("all-reduce.273_312", /* collective name */
                        2, 3, 1, 3, /* slice_src, tpu_src, slice_dst, tpu_dst */
                        /* start_timestamp_ns, end_timestamp_ns, duration_us */
                        50000, 100000, 50,
                        /* size_bytes, chunk_id, loop_index_id */
                        32768, 0, 24,
                        /* validity_info */
                        DCN_MESSAGE_VALID));
  EXPECT_THAT(dcn_events_processor.GetMessage(1),
              FieldsAre("super-collective.1234", /* collective name */
                        /* slice_src, tpu_src, slice_dst, tpu_dst */
                        112, 1, 34, 2,
                        /* start_timestamp_ns. end_timestamp_ns, duration_us */
                        125000, 175000, 50,
                        /* size_bytes, chunk_id, loop_index_id */
                        1, 4, 0,
                        /* validity_info */
                        DCN_MESSAGE_VALID));
  EXPECT_THAT(
      dcn_events_processor.GetMessage(2),
      FieldsAre("super-collective", /* collective name */
                9, 3, 0, 0,         /* slice_src, tpu_src, slice_dst, tpu_dst */
                75000, 150000,      /* start_timestamp_ns. end_timestamp_ns */
                75,                 /* duration_us */
                10, -1, -1,         /* size_bytes, chunk_id, loop_index_id */
                /* validity_info */
                DCN_MESSAGE_VALID));
  TimestampMap host_ts_map = dcn_events_processor.HostTsMap();
  ASSERT_EQ(host_ts_map.size(), 6);
  for (const auto &ts_map_item : host_ts_map) {
    ASSERT_EQ(ts_map_item.first, ts_map_item.second->timestamp_ns);
    if (ts_map_item.first == 50000) {
      ASSERT_EQ(ts_map_item.second->duration_ns, 0);
      ASSERT_EQ(ts_map_item.second->message_diff, 1);
      ASSERT_EQ(ts_map_item.second->size_diff, 32768);
    } else if (ts_map_item.first == 125000) {
      ASSERT_EQ(ts_map_item.second->duration_ns, 0);
      ASSERT_EQ(ts_map_item.second->message_diff, 1);
      ASSERT_EQ(ts_map_item.second->size_diff, 1);
    } else if (ts_map_item.first == 75000) {
      ASSERT_EQ(ts_map_item.second->duration_ns, 0);
      ASSERT_EQ(ts_map_item.second->message_diff, 1);
      ASSERT_EQ(ts_map_item.second->size_diff, 10);
    } else if (ts_map_item.first == 100000) {
      ASSERT_EQ(ts_map_item.second->duration_ns, 50000);
      ASSERT_EQ(ts_map_item.second->message_diff, -1);
      ASSERT_EQ(ts_map_item.second->size_diff, -32768);
    } else if (ts_map_item.first == 175000) {
      ASSERT_EQ(ts_map_item.second->duration_ns, 50000);
      ASSERT_EQ(ts_map_item.second->message_diff, -1);
      ASSERT_EQ(ts_map_item.second->size_diff, -1);
    } else if (ts_map_item.first == 150000) {
      ASSERT_EQ(ts_map_item.second->duration_ns, 75000);
      ASSERT_EQ(ts_map_item.second->message_diff, -1);
      ASSERT_EQ(ts_map_item.second->size_diff, -10);
    } else {
      FAIL() << "Unexpected timestamp entry.";
    }
  }
  const std::vector<DcnBurst> &host_bursts =
      dcn_events_processor.GetHostBursts();
  ASSERT_EQ(host_bursts.size(), 1);
  ASSERT_EQ(host_bursts[0].num_messages, 3);
  ASSERT_EQ(host_bursts[0].start_timestamp_ns, 50000);
  ASSERT_EQ(host_bursts[0].end_timestamp_ns, 175000);
  ASSERT_EQ(host_bursts[0].burst_size_bytes, 32779);
  ASSERT_EQ(host_bursts[0].max_overlapping_messages, 2);
}

// Loopback message test, currently interpreted as valid.
TEST(DcnAnalysis, CreateLoopBackMessageTest) {
  XSpace space;
  XPlane *host_trace = space.add_planes();
  XPlaneBuilder xplane_builder(host_trace);

  XEventMetadata *event_metadata_1 = xplane_builder.GetOrCreateEventMetadata(1);
  event_metadata_1->set_name(std::string(kMegaScaleDcnReceive));

  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventBuilder event_builder = xline_builder.AddEvent(*event_metadata_1);
  event_builder.SetOffsetNs(5000000);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_label"), "all-gather.1234");
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_slice_id"), 2);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_per_slice_device_id"),
      3);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_destination_slice_id"), 2);
  event_builder.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                                 "dcn_destination_per_slice_device_id"),
                             1);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_chunk"), 4);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_loop_index"), 40);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("duration_us"), 1000);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("payload_size_bytes"), 1000);
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_trace);
  DcnEventsProcessor dcn_events_processor(4, false);
  dcn_events_processor.SetupMessageInfo(plane);
  dcn_events_processor.ProcessReceiveMessages(plane);
  ASSERT_EQ(dcn_events_processor.NumReceivedMessages(), 1);
  EXPECT_THAT(dcn_events_processor.GetMessage(0),
              FieldsAre("all-gather.1234", /* collective name */
                        2, 3, 2, 1, /* slice_src, tpu_src, slice_dst, tpu_dst */
                        /* start_timestamp_ns. end_timestamp_ns, duration_us */
                        4000000, 5000000, 1000,
                        /* size_bytes, chunk_id, loop_index_id */
                        1000, 4, 40,
                        /* validity_info */
                        DCN_MESSAGE_VALID_LOOPBACK));
}

// Zero duration message, this is due to a bug or clock skew between source
// and destination. Any analysis will just cause confusion, mark it as invalid.
TEST(DcnAnalysis, CreateZeroDurationMessageTest) {
  XSpace space;
  XPlane *host_trace = space.add_planes();
  XPlaneBuilder xplane_builder(host_trace);

  XEventMetadata *event_metadata_1 = xplane_builder.GetOrCreateEventMetadata(1);
  event_metadata_1->set_name(std::string(kMegaScaleDcnReceive));

  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventBuilder event_builder = xline_builder.AddEvent(*event_metadata_1);
  event_builder.SetOffsetNs(20000);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_label"),
      "all-reduce.273_312");
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_slice_id"), 2);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_source_per_slice_device_id"),
      3);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_destination_slice_id"), 1);
  event_builder.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                                 "dcn_destination_per_slice_device_id"),
                             1);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_chunk"), 0);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("dcn_loop_index"), 25);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("duration_us"), 0);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("payload_size_bytes"), 512);
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_trace);
  DcnEventsProcessor dcn_events_processor(4, false);
  dcn_events_processor.SetupMessageInfo(plane);
  dcn_events_processor.ProcessReceiveMessages(plane);
  EXPECT_THAT(
      dcn_events_processor.GetMessage(0),
      FieldsAre("all-reduce.273_312", /* collective name */
                2, 3, 1, 1, /* slice_src, tpu_src, slice_dst, tpu_dst */
                20000, 20000,
                0, /* start_timestamp_ns. end_timestamp_ns, duration_us */
                512, 0, 25, /* size_bytes, chunk_id, loop_index_id */
                            /* validity_info */
                DCN_MESSAGE_INVALID_CLOCK_SKEW));
}

// Missing key test, make sure it is invalid and correctly initialized.
TEST(DcnAnalysis, CreateMissingKeyTest) {
  XSpace space;
  XPlane *host_trace = space.add_planes();
  XPlaneBuilder xplane_builder(host_trace);

  XEventMetadata *event_metadata_1 = xplane_builder.GetOrCreateEventMetadata(1);
  event_metadata_1->set_name(std::string(kMegaScaleDcnReceive));

  XLineBuilder xline_builder = xplane_builder.GetOrCreateLine(0);
  XEventBuilder event_builder = xline_builder.AddEvent(*event_metadata_1);
  event_builder.SetOffsetNs(50000);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("duration_us"), 10);
  event_builder.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata("payload_size_bytes"), 100);

  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_trace);
  DcnEventsProcessor dcn_events_processor(4, false);
  dcn_events_processor.SetupMessageInfo(plane);
  dcn_events_processor.ProcessReceiveMessages(plane);
  EXPECT_THAT(
      dcn_events_processor.GetMessage(0),
      FieldsAre("",             /* collective name */
                -1, -1, -1, -1, /* slice_src, tpu_src, slice_dst, tpu_dst */
                40000, 50000,   /* start_timestamp_ns. end_timestamp_ns, */
                10,             /* duration_us */
                100, -1, -1,    /* size_bytes, chunk_id, loop_index_id */
                                /* validity_info */
                DCN_MESSAGE_INVALID_BAD_KEY));
}

}  // namespace

}  // namespace profiler
}  // namespace tensorflow
