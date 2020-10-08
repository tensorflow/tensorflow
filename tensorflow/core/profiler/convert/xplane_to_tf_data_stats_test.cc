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

#include "tensorflow/core/profiler/convert/xplane_to_tf_data_stats.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/tf_data_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::EqualsProto;

// Test with the following example dataset:
// dataset = tf.data.Dataset.range(8)
// dataset = dataset.prefetch(2)
// for _ in dataset:
//   pass
TEST(XPlaneToTfDataStatsTest, HostInputPipeline) {
  constexpr int64 kPrefetchIteratorId = 123;
  constexpr int64 kRangeIteratorId = 456;
  constexpr int64 kFirstElementId = 100;
  constexpr int64 kSecondElementId = 200;

  XPlane host_plane;
  XPlaneBuilder host_plane_builder(&host_plane);
  host_plane_builder.ReserveLines(2);

  auto consumer_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::Prefetch", 0,
               100, {{StatType::kStepId, kPrefetchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread,
               HostEventType::kPrefetchConsume, 80, 20,
               {{StatType::kElementId, kFirstElementId}});
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::Prefetch", 200,
               20, {{StatType::kStepId, kPrefetchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread,
               HostEventType::kPrefetchConsume, 210, 10,
               {{StatType::kElementId, kSecondElementId}});

  auto producer_thread = host_plane_builder.GetOrCreateLine(1);
  // Blocking producer.
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kPrefetchProduce, 0, 80,
               {{StatType::kElementId, kFirstElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::Prefetch::Range", 0, 80,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kPrefetchIteratorId}});
  // Non-blocking producer.
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kPrefetchProduce, 100, 80,
               {{StatType::kElementId, kSecondElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::Prefetch::Range", 100, 80,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kPrefetchIteratorId}});

  TfDataStats tf_data_stats = ConvertXPlaneToTfDataStats(&host_plane);
  EXPECT_THAT(tf_data_stats, EqualsProto(R"pb(
                iterator_metadata: {
                  key: 123,
                  value: {
                    id: 123
                    name: "Prefetch"
                    long_name: "Iterator::Prefetch"
                    is_async: true
                  }
                }
                iterator_metadata: {
                  key: 456,
                  value: {
                    id: 456
                    parent_id: 123
                    name: "Range"
                    long_name: "Iterator::Prefetch::Range"
                    is_async: false
                  }
                }
                input_pipelines {
                  key: 123,
                  value: {
                    metadata { id: 123 type: HOST name: "Host:0" }
                    avg_latency_ps: 60
                    min_latency_ps: 20
                    max_latency_ps: 100
                    stats {
                      bottleneck_iterator_id: 456
                      iterator_stats {
                        key: 123,
                        value: {
                          id: 123
                          start_time_ps: 0
                          duration_ps: 100
                          self_time_ps: 20
                          is_blocking: true
                          num_calls: 1
                        }
                      }
                      iterator_stats {
                        key: 456,
                        value: {
                          id: 456
                          start_time_ps: 0
                          duration_ps: 80
                          self_time_ps: 80
                          is_blocking: true
                          num_calls: 1
                        }
                      }
                    }
                    stats {
                      bottleneck_iterator_id: 123
                      iterator_stats {
                        key: 123,
                        value: {
                          id: 123
                          start_time_ps: 200
                          duration_ps: 20
                          self_time_ps: 20
                          is_blocking: true
                          num_calls: 1
                        }
                      }
                      iterator_stats {
                        key: 456,
                        value: {
                          id: 456
                          start_time_ps: 100
                          duration_ps: 80
                          self_time_ps: 80
                          is_blocking: false
                          num_calls: 1
                        }
                      }
                    }
                  }
                }
              )pb"));
}

TEST(XPlaneToTfDataStatsTest, DeviceInputPipeline) {
  constexpr int64 kPrefetchIteratorId = 123;
  constexpr int64 kRangeIteratorId = 456;
  constexpr int64 kElementId = 100;

  XPlane host_plane;
  XPlaneBuilder host_plane_builder(&host_plane);
  host_plane_builder.ReserveLines(2);

  auto consumer_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::Prefetch", 0,
               30, {{StatType::kStepId, kPrefetchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::Prefetch", 100,
               100, {{StatType::kStepId, kPrefetchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread,
               HostEventType::kPrefetchConsume, 180, 20,
               {{StatType::kElementId, kElementId}});

  auto producer_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kPrefetchProduce, 100, 80,
               {{StatType::kElementId, kElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::Prefetch::Generator", 100, 80,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kPrefetchIteratorId}});

  TfDataStats tf_data_stats = ConvertXPlaneToTfDataStats(&host_plane);
  EXPECT_THAT(tf_data_stats, EqualsProto(R"pb(
                iterator_metadata: {
                  key: 123,
                  value: {
                    id: 123
                    name: "Prefetch"
                    long_name: "Iterator::Prefetch"
                    is_async: true
                  }
                }
                iterator_metadata: {
                  key: 456,
                  value: {
                    id: 456
                    parent_id: 123
                    name: "Generator"
                    long_name: "Iterator::Prefetch::Generator"
                    is_async: false
                  }
                }
                input_pipelines {
                  key: 123,
                  value: {
                    metadata { id: 123 type: DEVICE name: "Device:0" }
                    avg_latency_ps: 65
                    min_latency_ps: 30
                    max_latency_ps: 100
                    stats {
                      bottleneck_iterator_id: 456
                      iterator_stats {
                        key: 123,
                        value: {
                          id: 123
                          start_time_ps: 100
                          duration_ps: 100
                          self_time_ps: 20
                          is_blocking: true
                          num_calls: 1
                        }
                      }
                      iterator_stats {
                        key: 456,
                        value: {
                          id: 456
                          start_time_ps: 100
                          duration_ps: 80
                          self_time_ps: 80
                          is_blocking: true
                          num_calls: 1
                        }
                      }
                    }
                    stats {
                      bottleneck_iterator_id: 123
                      iterator_stats {
                        key: 123,
                        value: {
                          id: 123
                          start_time_ps: 0
                          duration_ps: 30
                          self_time_ps: 30
                          is_blocking: true
                          num_calls: 1
                        }
                      }
                    }
                  }
                }
              )pb"));
}

// Test with the following example dataset:
// dataset = tf.data.Dataset.range(8)
// dataset = dataset.map(lambda x: x + 1)
// dataset = dataset.batch(2)
// for _ in dataset:
//   pass
TEST(XPlaneToTfDataStatsTest, MapAndBatch) {
  constexpr int64 kMapAndBatchIteratorId = 123;
  constexpr int64 kRangeIteratorId = 456;
  constexpr int64 kElementId = 100;

  XPlane host_plane;
  XPlaneBuilder host_plane_builder(&host_plane);
  host_plane_builder.ReserveLines(2);

  XLineBuilder consumer_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::MapAndBatch",
               0, 100, {{StatType::kStepId, kMapAndBatchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread,
               HostEventType::kMapAndBatchConsume, 80, 20,
               {{StatType::kElementId, kElementId}});

  XLineBuilder producer_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kMapAndBatchProduce, 0, 30,
               {{StatType::kElementId, kElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::MapAndBatch::Range", 0, 30,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kMapAndBatchIteratorId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kMapAndBatchProduce, 40, 30,
               {{StatType::kElementId, kElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::MapAndBatch::Range", 40, 30,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kMapAndBatchIteratorId}});

  TfDataStats tf_data_stats = ConvertXPlaneToTfDataStats(&host_plane);
  EXPECT_THAT(tf_data_stats, EqualsProto(R"pb(
                iterator_metadata: {
                  key: 123,
                  value: {
                    id: 123
                    name: "MapAndBatch"
                    long_name: "Iterator::MapAndBatch"
                    is_async: true
                  }
                }
                iterator_metadata: {
                  key: 456,
                  value: {
                    id: 456
                    parent_id: 123
                    name: "Range"
                    long_name: "Iterator::MapAndBatch::Range"
                    is_async: false
                  }
                }
                input_pipelines {
                  key: 123,
                  value: {
                    metadata { id: 123 type: HOST name: "Host:0" }
                    avg_latency_ps: 100
                    min_latency_ps: 100
                    max_latency_ps: 100
                    stats {
                      bottleneck_iterator_id: 456
                      iterator_stats {
                        key: 123,
                        value: {
                          id: 123
                          start_time_ps: 0
                          duration_ps: 100
                          self_time_ps: 40
                          is_blocking: true
                          num_calls: 1
                        }
                      }
                      iterator_stats {
                        key: 456,
                        value: {
                          id: 456
                          start_time_ps: 0
                          duration_ps: 60
                          self_time_ps: 60
                          is_blocking: true
                          num_calls: 2
                        }
                      }
                    }
                  }
                }
              )pb"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
