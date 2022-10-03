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
#include "tensorflow/core/profiler/convert/xplane_to_profile_response.h"

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/protobuf/input_pipeline.pb.h"
#include "tensorflow/core/profiler/protobuf/op_profile.pb.h"
#include "tensorflow/core/profiler/protobuf/overview_page.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Property;

void CreateXSpace(XSpace* space) {
  constexpr double kDevCapPeakTeraflopsPerSecond = 141.0;
  constexpr double kDevCapPeakHbmBwGigabytesPerSecond = 900.0;

  XPlaneBuilder host_plane(space->add_planes());
  XPlaneBuilder device_plane(space->add_planes());

  host_plane.SetName("cpu");
  host_plane.SetId(0);
  XLineBuilder thread1 = host_plane.GetOrCreateLine(10);
  thread1.SetName("thread1");
  XEventBuilder event1 =
      thread1.AddEvent(*host_plane.GetOrCreateEventMetadata("event1"));
  event1.SetTimestampNs(150000);
  event1.SetDurationNs(10000);
  event1.AddStatValue(*host_plane.GetOrCreateStatMetadata("tf_op"),
                      *host_plane.GetOrCreateStatMetadata("Relu"));
  event1.AddStatValue(*host_plane.GetOrCreateStatMetadata(GetStatTypeStr(
                          StatType::kDevCapPeakTeraflopsPerSecond)),
                      kDevCapPeakTeraflopsPerSecond);
  event1.AddStatValue(*host_plane.GetOrCreateStatMetadata(GetStatTypeStr(
                          StatType::kDevCapPeakHbmBwGigabytesPerSecond)),
                      kDevCapPeakHbmBwGigabytesPerSecond);
  XLineBuilder thread2 = host_plane.GetOrCreateLine(20);
  thread2.SetName("thread2");
  XEventBuilder event2 =
      thread2.AddEvent(*host_plane.GetOrCreateEventMetadata("event2"));
  event2.SetTimestampNs(160000);
  event2.SetDurationNs(10000);
  event2.AddStatValue(*host_plane.GetOrCreateStatMetadata("tf_op"),
                      *host_plane.GetOrCreateStatMetadata("Conv2D"));
  event2.AddStatValue(*host_plane.GetOrCreateStatMetadata(GetStatTypeStr(
                          StatType::kDevCapPeakTeraflopsPerSecond)),
                      kDevCapPeakTeraflopsPerSecond);
  event2.AddStatValue(*host_plane.GetOrCreateStatMetadata(GetStatTypeStr(
                          StatType::kDevCapPeakHbmBwGigabytesPerSecond)),
                      kDevCapPeakHbmBwGigabytesPerSecond);

  device_plane.SetName("gpu:0");
  device_plane.SetId(1);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(30);
  stream1.SetName("gpu stream 1");
  XEventBuilder event3 =
      stream1.AddEvent(*device_plane.GetOrCreateEventMetadata("kernel1"));
  event3.SetTimestampNs(180000);
  event3.SetDurationNs(10000);
  event3.AddStatValue(*device_plane.GetOrCreateStatMetadata("correlation id"),
                      55);
  event3.AddStatValue(*host_plane.GetOrCreateStatMetadata(GetStatTypeStr(
                          StatType::kDevCapPeakTeraflopsPerSecond)),
                      kDevCapPeakTeraflopsPerSecond);
  event3.AddStatValue(*host_plane.GetOrCreateStatMetadata(GetStatTypeStr(
                          StatType::kDevCapPeakHbmBwGigabytesPerSecond)),
                      kDevCapPeakHbmBwGigabytesPerSecond);
}

TEST(ConvertXPlaneToProfileResponse, ExtractTpuMxuUtilizationFromXSpace) {
  constexpr double kDevCapPeakTeraflopsPerSecond = 141.0;
  constexpr double kDevCapPeakHbmBwGigabytesPerSecond = 900.0;

  XSpace xspace;
  XPlaneBuilder devicePlane(
      GetOrCreateTpuXPlane(&xspace, 0, "TPU V4", kDevCapPeakTeraflopsPerSecond,
                           kDevCapPeakHbmBwGigabytesPerSecond));
  auto xplane = FindOrAddMutablePlaneWithName(&xspace, kHostThreadsPlaneName);
  XPlaneBuilder xplaneBuilder(xplane);
  xplaneBuilder.ReserveLines(1);
  xplaneBuilder.AddStatValue(
      *xplaneBuilder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kMatrixUnitUtilizationPercent)),
      20.0);
  devicePlane.AddStatValue(*devicePlane.GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kDevCapCoreCount)),
                           80);
  ProfileRequest request;
  request.add_tools("overview_page");
  ProfileResponse response;
  TF_CHECK_OK(ConvertXSpaceToProfileResponse(xspace, request, &response));
  EXPECT_EQ(1, response.tool_data_size());
  EXPECT_EQ("overview_page.pb", response.tool_data(0).name());
  OverviewPage overview_page;
  ASSERT_TRUE(overview_page.ParseFromString(response.tool_data(0).data()));
  EXPECT_EQ(overview_page.analysis().mxu_utilization_percent(), 20);
  EXPECT_EQ(overview_page.run_environment().device_type(), "TPU V4");
  EXPECT_EQ(overview_page.run_environment().device_core_count(), 1);
}

TEST(ConvertXPlaneToProfileResponse, TraceViewer) {
  XSpace xspace;
  CreateXSpace(&xspace);
  ProfileRequest request;
  ProfileResponse response;
  TF_CHECK_OK(ConvertXSpaceToProfileResponse(xspace, request, &response));
}

TEST(ConvertXPlaneToProfileResponse, OverviewPage) {
  XSpace xspace;
  CreateXSpace(&xspace);
  ProfileRequest request;
  request.add_tools("overview_page");
  ProfileResponse response;
  TF_CHECK_OK(ConvertXSpaceToProfileResponse(xspace, request, &response));
  EXPECT_EQ(1, response.tool_data_size());
  EXPECT_EQ("overview_page.pb", response.tool_data(0).name());
  OverviewPage overview_page;
  ASSERT_TRUE(overview_page.ParseFromString(response.tool_data(0).data()));
}

TEST(ConvertXPlaneToProfileResponse, InputPipeline) {
  XSpace xspace;
  CreateXSpace(&xspace);
  ProfileRequest request;
  request.add_tools("input_pipeline");
  ProfileResponse response;
  TF_CHECK_OK(ConvertXSpaceToProfileResponse(xspace, request, &response));
  EXPECT_EQ(1, response.tool_data_size());
  EXPECT_EQ("input_pipeline.pb", response.tool_data(0).name());
  InputPipelineAnalysisResult input_pipeline;
  ASSERT_TRUE(input_pipeline.ParseFromString(response.tool_data(0).data()));
}

TEST(ConvertXPlaneToProfileResponse, TensorflowStats) {
  XSpace xspace;
  CreateXSpace(&xspace);
  ProfileRequest request;
  request.add_tools("tensorflow_stats");
  ProfileResponse response;
  TF_CHECK_OK(ConvertXSpaceToProfileResponse(xspace, request, &response));
  EXPECT_EQ(1, response.tool_data_size());
  EXPECT_EQ("tensorflow_stats.pb", response.tool_data(0).name());
  TfStatsDatabase tf_stats_db;
  ASSERT_TRUE(tf_stats_db.ParseFromString(response.tool_data(0).data()));
}

TEST(ConvertXPlaneToProfileResponse, XPlane) {
  XSpace xspace;
  CreateXSpace(&xspace);
  ProfileRequest request;
  request.add_tools("xplane.pb");
  ProfileResponse response;
  TF_CHECK_OK(ConvertXSpaceToProfileResponse(xspace, request, &response));
  EXPECT_EQ(1, response.tool_data_size());
  EXPECT_EQ("xplane.pb", response.tool_data(0).name());
  ASSERT_TRUE(xspace.ParseFromString(response.tool_data(0).data()));
}

TEST(ConvertXPlaneToProfileResponse, OpProfile) {
  XSpace xspace;
  CreateXSpace(&xspace);
  ProfileRequest request;
  request.add_tools("op_profile");
  ProfileResponse response;
  TF_CHECK_OK(ConvertXSpaceToProfileResponse(xspace, request, &response));
  EXPECT_EQ(1, response.tool_data_size());
  EXPECT_THAT(
      response.tool_data(),
      UnorderedElementsAre(Property(&ProfileToolData::name, "op_profile.pb")));
  tensorflow::profiler::op_profile::Profile op_profile;
  ASSERT_TRUE(op_profile.ParseFromString(response.tool_data(0).data()));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
