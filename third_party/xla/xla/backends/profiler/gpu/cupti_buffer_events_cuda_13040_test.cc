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

// Tests for the CUDA 13.4+ activity record conversion. Built only with CUDA
// 13.4+ headers (see the cupti_buffer_events_test BUILD target).

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_result.h"
#include "xla/backends/profiler/gpu/cupti_buffer_events.h"
#include "xla/backends/profiler/gpu/mock_cupti.h"

namespace xla {
namespace profiler {
namespace test {
namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

// Converts CUDA 13.4 activity records from CUDA graph nodes, with MockCupti in
// place of libcupti.
class Cuda1304ActivityRecordsTest : public ::testing::Test {
 protected:
  Cuda1304ActivityRecordsTest() {
    kernel_record_.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    kernel_record_.name = "test_kernel";
    kernel_record_.start = 100;
    kernel_record_.end = 200;
    kernel_record_.deviceId = 0;
    kernel_record_.contextId = 1;
    kernel_record_.streamId = 2;
    kernel_record_.correlationId = 10;
    kernel_record_.graphId = 50;
    kernel_record_.graphNodeId = 500;
    kernel_record_.sourceGraphId = 7;
    kernel_record_.sourceGraphNodeId = (uint64_t{7} << 32) | 3;

    memcpy_record_.kind = CUPTI_ACTIVITY_KIND_MEMCPY;
    memcpy_record_.copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOD;
    memcpy_record_.bytes = 1024;
    memcpy_record_.start = 210;
    memcpy_record_.end = 250;
    memcpy_record_.deviceId = 0;
    memcpy_record_.contextId = 1;
    memcpy_record_.streamId = 2;
    memcpy_record_.correlationId = 10;
    memcpy_record_.graphId = 50;
    memcpy_record_.graphNodeId = 501;
    memcpy_record_.sourceGraphId = 7;
    memcpy_record_.sourceGraphNodeId = (uint64_t{7} << 32) | 4;

    memset_record_.kind = CUPTI_ACTIVITY_KIND_MEMSET;
    memset_record_.bytes = 512;
    memset_record_.memoryKind = CUPTI_ACTIVITY_MEMORY_KIND_DEVICE;
    memset_record_.start = 260;
    memset_record_.end = 300;
    memset_record_.deviceId = 0;
    memset_record_.contextId = 1;
    memset_record_.streamId = 2;
    memset_record_.correlationId = 10;
    memset_record_.graphId = 50;
    memset_record_.graphNodeId = 502;
    memset_record_.sourceGraphId = 8;
    memset_record_.sourceGraphNodeId = (uint64_t{8} << 32) | 1;

    p2p_record_.kind = CUPTI_ACTIVITY_KIND_MEMCPY2;
    p2p_record_.bytes = 256;
    p2p_record_.start = 310;
    p2p_record_.end = 350;
    p2p_record_.srcDeviceId = 0;
    p2p_record_.dstDeviceId = 1;
    p2p_record_.contextId = 1;
    p2p_record_.streamId = 2;
    p2p_record_.correlationId = 10;
    p2p_record_.graphId = 50;
    p2p_record_.graphNodeId = 503;
  }

  // Runs `records` through AddActivityBufferListEventsTo() and returns the
  // resulting events in record order.
  std::vector<CuptiTracerEvent> ConvertRecords(
      const std::vector<void*>& records) {
    auto& next_record =
        EXPECT_CALL(mock_cupti_, ActivityGetNextRecord(_, _, _));
    for (void* record : records) {
      next_record.WillOnce(
          DoAll(SetArgPointee<2>(static_cast<CUpti_Activity*>(record)),
                Return(CUPTI_SUCCESS)));
    }
    next_record.WillOnce(Return(CUPTI_ERROR_MAX_LIMIT_REACHED));

    std::vector<CuptiTracerEvent> events;
    CuptiEventCollectorDelegate delegate(
        annotation_map_,
        [&](CuptiTracerEvent&& event) { events.push_back(std::move(event)); });
    CuptiActivityBufferManager::CachedActivityBufferBatch batch;
    batch.buffers.emplace_back(/*p=*/nullptr, /*sz=*/128);
    size_t dropped = 0;
    AddActivityBufferListEventsTo(delegate, batch,
                                  /*max_activity_event_count=*/10, dropped,
                                  &mock_cupti_);
    return events;
  }

  CUpti_ActivityKernel13 kernel_record_{};
  CUpti_ActivityMemcpy7 memcpy_record_{};
  CUpti_ActivityMemset5 memset_record_{};
  CUpti_ActivityMemcpyPtoP4 p2p_record_{};
  MockCupti mock_cupti_;
  AnnotationMap annotation_map_{/*max_size=*/100, /*num_gpus=*/1};
};

TEST_F(Cuda1304ActivityRecordsTest,
       ExtractsSourceGraphIdFromCuda1304ActivityRecords) {
  std::vector<CuptiTracerEvent> events =
      ConvertRecords({&kernel_record_, &memcpy_record_, &memset_record_});

  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(events[0].type, CuptiTracerEventType::Kernel);
  EXPECT_EQ(events[0].graph_id, 50);
  EXPECT_EQ(events[0].graph_node_id, 500);
  EXPECT_EQ(events[0].source_graph_id, 7);
  EXPECT_EQ(events[0].source_graph_node_id, (uint64_t{7} << 32) | 3);

  EXPECT_EQ(events[1].type, CuptiTracerEventType::MemcpyD2D);
  EXPECT_EQ(events[1].graph_id, 50);
  EXPECT_EQ(events[1].graph_node_id, 501);
  EXPECT_EQ(events[1].source_graph_id, 7);
  EXPECT_EQ(events[1].source_graph_node_id, (uint64_t{7} << 32) | 4);

  EXPECT_EQ(events[2].type, CuptiTracerEventType::Memset);
  EXPECT_EQ(events[2].graph_id, 50);
  EXPECT_EQ(events[2].graph_node_id, 502);
  EXPECT_EQ(events[2].source_graph_id, 8);
  EXPECT_EQ(events[2].source_graph_node_id, (uint64_t{8} << 32) | 1);
}

// CUDA 13.4 P2P memcpy records have no source graph fields, so their events
// keep source_graph_id 0.
TEST_F(Cuda1304ActivityRecordsTest, P2PMemcpyHasNoSourceGraphId) {
  std::vector<CuptiTracerEvent> events = ConvertRecords({&p2p_record_});

  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].type, CuptiTracerEventType::MemcpyP2P);
  EXPECT_EQ(events[0].graph_id, 50);
  EXPECT_EQ(events[0].graph_node_id, 503);
  EXPECT_EQ(events[0].source_graph_id, 0);
  EXPECT_EQ(events[0].source_graph_node_id, 0);
}

}  // namespace
}  // namespace test
}  // namespace profiler
}  // namespace xla
