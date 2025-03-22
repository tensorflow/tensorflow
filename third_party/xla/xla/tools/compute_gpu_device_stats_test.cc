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

#include "xla/tools/compute_gpu_device_stats.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "third_party/tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "third_party/tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "third_party/tensorflow/core/profiler/utils/xplane_builder.h"
#include "third_party/tensorflow/core/profiler/utils/xplane_schema.h"
#include "third_party/tensorflow/core/profiler/utils/xplane_test_utils.h"

using tensorflow::profiler::StatType;

namespace xla::gpu {
namespace {

// Stat metadata IDs for clarity and maintainability.
constexpr int64_t kMemcpyDetailsMetadataId = 1;
constexpr int64_t kOtherDetailsMetadataId = 2;

// Test fixture for ComputeGpuDeviceStatsTest to reduce verbosity.
class ComputeGpuDeviceStatsTest : public testing::Test {
 protected:
  tensorflow::profiler::XSpace xspace_;
  tensorflow::profiler::XPlane* device_plane_;
  tensorflow::profiler::XPlane* host_plane_;

  ComputeGpuDeviceStatsTest() = default;

  void SetUp() override {
    device_plane_ = xspace_.add_planes();
    device_plane_->set_name("/device:GPU:0");
    host_plane_ = tensorflow::profiler::GetOrCreateHostXPlane(&xspace_);
    AddMemcpyStatMetadata();
    SetupHostPlaneBuilder();
  }

  void AddMemcpyStatMetadata() {
    tensorflow::profiler::XStatMetadata* stat_metadata =
        &(*device_plane_->mutable_stat_metadata())[kMemcpyDetailsMetadataId];
    stat_metadata->set_id(kMemcpyDetailsMetadataId);
    stat_metadata->set_name("memcpy_details");
  }

  void SetupHostPlaneBuilder() {
    host_plane_builder_ =
        std::make_unique<tensorflow::profiler::XPlaneBuilder>(host_plane_);
    host_plane_builder_->ReserveLines(1);
    tf_executor_thread_ = std::make_unique<tensorflow::profiler::XLineBuilder>(
        host_plane_builder_->GetOrCreateLine(0));
  }

  void AddDeviceStat(absl::string_view stat_name, absl::string_view value) {
    device_plane_builder_ =
        std::make_unique<tensorflow::profiler::XPlaneBuilder>(device_plane_);
    device_plane_builder_->AddStatValue(
        *device_plane_builder_->GetOrCreateStatMetadata(stat_name), value);
  }

  void AddDeviceStat(absl::string_view stat_name, uint64_t value) {
    device_plane_builder_ =
        std::make_unique<tensorflow::profiler::XPlaneBuilder>(device_plane_);
    device_plane_builder_->AddStatValue(
        *device_plane_builder_->GetOrCreateStatMetadata(stat_name), value);
  }

  tensorflow::profiler::XLine* AddLineToDevicePlane() {
    return device_plane_->add_lines();
  }

  tensorflow::profiler::XEvent* AddEventToLine(
      tensorflow::profiler::XLine* line, int64_t duration_ps,
      int64_t metadata_id) {
    tensorflow::profiler::XEvent* event = line->add_events();
    event->set_duration_ps(duration_ps);
    event->add_stats()->set_metadata_id(metadata_id);
    return event;
  }

  void AddMemoryAllocationEvent(int64_t start_timestamp_ps) {
    tensorflow::profiler::CreateXEvent(
        host_plane_builder_.get(), tf_executor_thread_.get(),
        "MemoryAllocation", start_timestamp_ps, 1000,
        {{StatType::kBytesReserved, int64_t{2000}},
         {StatType::kBytesAllocated, int64_t{3000}},
         {StatType::kBytesAvailable, int64_t{5000}},
         {StatType::kPeakBytesInUse, int64_t{8500}},
         {StatType::kRequestedBytes, int64_t{200}},
         {StatType::kAllocationBytes, int64_t{256}},
         {StatType::kAddress, int64_t{222333}},
         {StatType::kStepId, int64_t{-93746}},
         {StatType::kDataType, int64_t{1}},
         {StatType::kAllocatorName, "GPU_0_bfc"},
         {StatType::kTfOp, "foo/bar"},
         {StatType::kRegionType, "output"},
         {StatType::kTensorShapes, "[3, 3, 512, 512]"}});
  }

  void AddMemoryDeallocationEvent(int64_t start_timestamp_ps) {
    tensorflow::profiler::CreateXEvent(
        host_plane_builder_.get(), tf_executor_thread_.get(),
        "MemoryDeallocation", start_timestamp_ps, 1000,
        {{StatType::kBytesReserved, int64_t{2000}},
         {StatType::kBytesAllocated, int64_t{2744}},
         {StatType::kBytesAvailable, int64_t{5256}},
         {StatType::kPeakBytesInUse, int64_t{8500}},
         {StatType::kRequestedBytes, int64_t{200}},
         {StatType::kAllocationBytes, int64_t{256}},
         {StatType::kAddress, int64_t{222333}},
         {StatType::kStepId, int64_t{0}},
         {StatType::kDataType, int64_t{0}},
         {StatType::kAllocatorName, "GPU_0_bfc"},
         {StatType::kRegionType, ""},
         {StatType::kTensorShapes, ""}});
  }

  std::unique_ptr<tensorflow::profiler::XPlaneBuilder> host_plane_builder_;
  std::unique_ptr<tensorflow::profiler::XLineBuilder> tf_executor_thread_;
  std::unique_ptr<tensorflow::profiler::XPlaneBuilder> device_plane_builder_;
};

TEST_F(ComputeGpuDeviceStatsTest, IsMemcpyTest) {
  tensorflow::profiler::XEvent event;
  event.add_stats()->set_metadata_id(kMemcpyDetailsMetadataId);
  EXPECT_TRUE(IsMemcpy(event, kMemcpyDetailsMetadataId));
  EXPECT_FALSE(IsMemcpy(event, kOtherDetailsMetadataId));
}

TEST_F(ComputeGpuDeviceStatsTest, ProcessLineEventsTest) {
  tensorflow::profiler::XLine line;
  AddEventToLine(&line, 1000, kMemcpyDetailsMetadataId);
  AddEventToLine(&line, 2000, kOtherDetailsMetadataId);
  AddEventToLine(&line, 3000, kMemcpyDetailsMetadataId);

  TF_ASSERT_OK_AND_ASSIGN(LineStats stats,
                          ProcessLineEvents(line, kMemcpyDetailsMetadataId));
  EXPECT_EQ(stats.total_time_ps, 6000);
  EXPECT_EQ(stats.memcpy_time_ps, 4000);
}

TEST_F(ComputeGpuDeviceStatsTest, CalculateDeviceTimeAndMemcpyTest) {
  tensorflow::profiler::XLine* line = AddLineToDevicePlane();
  tensorflow::profiler::XEvent* event =
      AddEventToLine(line, 1000, kMemcpyDetailsMetadataId);
  AddEventToLine(line, 2000, kOtherDetailsMetadataId);
  AddEventToLine(line, 3000, kMemcpyDetailsMetadataId);

  AddMemoryAllocationEvent(40000);
  AddMemoryDeallocationEvent(50000);

  // Device properties setup
  constexpr int kClockRateKHz = 1530000;
  constexpr int kCoreCount = 80;
  constexpr int64_t kMemoryBandwidthBytesPerSecond = 900000000000;
  constexpr int kComputeCapMajor = 7;
  constexpr int kComputeCapMinor = 0;

  AddDeviceStat(GetStatTypeStr(StatType::kDevVendor),
                tensorflow::profiler::kDeviceVendorNvidia);
  AddDeviceStat("clock_rate", kClockRateKHz);
  AddDeviceStat("core_count", kCoreCount);
  AddDeviceStat("memory_bandwidth", kMemoryBandwidthBytesPerSecond);
  AddDeviceStat("compute_cap_major", kComputeCapMajor);
  AddDeviceStat("compute_cap_minor", kComputeCapMinor);

  tsl::profiler::GroupTfEvents(&xspace_);
  TF_ASSERT_OK_AND_ASSIGN(GpuDeviceStats stats, ComputeGPUDeviceStats(xspace_));
  EXPECT_EQ(stats.device_time_us, 0.006);
  EXPECT_EQ(stats.device_memcpy_time_us, 0.004);
  EXPECT_TRUE(IsMemcpy(*event, kMemcpyDetailsMetadataId));
  EXPECT_FALSE(IsMemcpy(*event, kOtherDetailsMetadataId));
  EXPECT_EQ(stats.peak_device_mem_bytes, 8500);
}

TEST_F(ComputeGpuDeviceStatsTest, NoDevicePlaneReturnsErrorTest) {
  device_plane_->set_name("/device:CPU:0");
  EXPECT_TRUE(absl::IsNotFound(ComputeGPUDeviceStats(xspace_).status()));
}

}  // namespace
}  // namespace xla::gpu
