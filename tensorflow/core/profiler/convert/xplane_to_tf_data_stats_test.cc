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
  constexpr int64_t kPrefetchIteratorId = 123;
  constexpr int64_t kRangeIteratorId = 456;
  constexpr int64_t kFirstElementId = 100;
  constexpr int64_t kSecondElementId = 200;

  XPlane host_plane;
  XPlaneBuilder host_plane_builder(&host_plane);
  host_plane_builder.ReserveLines(2);

  auto consumer_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::Prefetch", 0,
               100000000, {{StatType::kStepId, kPrefetchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread,
               HostEventType::kPrefetchConsume, 80000000, 20000000,
               {{StatType::kElementId, kFirstElementId}});
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::Prefetch",
               200000000, 20000000, {{StatType::kStepId, kPrefetchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread,
               HostEventType::kPrefetchConsume, 210000000, 10000000,
               {{StatType::kElementId, kSecondElementId}});

  auto producer_thread = host_plane_builder.GetOrCreateLine(1);
  // Blocking producer.
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kPrefetchProduce, 0, 80000000,
               {{StatType::kElementId, kFirstElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::Prefetch::Range", 0, 80000000,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kPrefetchIteratorId}});
  // Non-blocking producer.
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kPrefetchProduce, 100000000, 80000000,
               {{StatType::kElementId, kSecondElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::Prefetch::Range", 100000000, 80000000,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kPrefetchIteratorId}});

  CombinedTfDataStats combined_tf_data_stats;
  CombinedTfDataStatsBuilder builder(&combined_tf_data_stats);
  builder.Add("host1", &host_plane);
  builder.Finalize();
  EXPECT_THAT(
      combined_tf_data_stats, EqualsProto(R"pb(
        bottleneck_analysis: {
          host: "host1"
          input_pipeline: "Host:0"
          max_latency_ps: 100000000
          iterator_name: "Range"
          iterator_long_name: "Iterator::Prefetch::Range"
          iterator_latency_ps: 80000000
          suggestion: "See <a href=\"https://www.tensorflow.org/guide/data_performance_analysis\" target=\"_blank\">this</a> for suggestions."
        }
        tf_data_stats: {
          key: "host1"
          value: {
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
                avg_latency_ps: 60000000
                min_latency_ps: 20000000
                max_latency_ps: 100000000
                num_slow_calls: 1
                stats {
                  bottleneck_iterator_id: 456
                  bottleneck_iterator_latency_ps: 80000000
                  iterator_stats {
                    key: 123,
                    value: {
                      id: 123
                      start_time_ps: 0
                      duration_ps: 100000000
                      self_time_ps: 20000000
                      is_blocking: true
                      num_calls: 1
                    }
                  }
                  iterator_stats {
                    key: 456,
                    value: {
                      id: 456
                      start_time_ps: 0
                      duration_ps: 80000000
                      self_time_ps: 80000000
                      is_blocking: true
                      num_calls: 1
                    }
                  }
                }
                stats {
                  bottleneck_iterator_id: 123
                  bottleneck_iterator_latency_ps: 20000000
                  iterator_stats {
                    key: 123,
                    value: {
                      id: 123
                      start_time_ps: 200000000
                      duration_ps: 20000000
                      self_time_ps: 20000000
                      is_blocking: true
                      num_calls: 1
                    }
                  }
                  iterator_stats {
                    key: 456,
                    value: {
                      id: 456
                      start_time_ps: 100000000
                      duration_ps: 80000000
                      self_time_ps: 80000000
                      is_blocking: false
                      num_calls: 1
                    }
                  }
                }
              }
            }
          }
        }
        is_input_bound: true
        summary: "Your profile has a tf.data input pipeline slower than 50 us. For each slow input pipeline, below shows a bottleneck in the input pipeline and a suggestion on how to fix it."
      )pb"));
}

TEST(XPlaneToTfDataStatsTest, DeviceInputPipeline) {
  constexpr int64_t kPrefetchIteratorId = 123;
  constexpr int64_t kRangeIteratorId = 456;
  constexpr int64_t kElementId = 100;

  XPlane host_plane;
  XPlaneBuilder host_plane_builder(&host_plane);
  host_plane_builder.ReserveLines(2);

  auto consumer_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::Prefetch", 0,
               30000000, {{StatType::kStepId, kPrefetchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::Prefetch",
               100000000, 100000000,
               {{StatType::kStepId, kPrefetchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread,
               HostEventType::kPrefetchConsume, 180000000, 20000000,
               {{StatType::kElementId, kElementId}});

  auto producer_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kPrefetchProduce, 100000000, 80000000,
               {{StatType::kElementId, kElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::Prefetch::Generator", 100000000, 80000000,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kPrefetchIteratorId}});

  CombinedTfDataStats combined_tf_data_stats;
  CombinedTfDataStatsBuilder builder(&combined_tf_data_stats);
  builder.Add("host1", &host_plane);
  builder.Finalize();
  // Device input pipeline is not considered for bottleneck analysis.
  EXPECT_THAT(
      combined_tf_data_stats, EqualsProto(R"pb(
        tf_data_stats: {
          key: "host1"
          value: {
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
                avg_latency_ps: 65000000
                min_latency_ps: 30000000
                max_latency_ps: 100000000
                num_slow_calls: 1
                stats {
                  bottleneck_iterator_id: 456
                  bottleneck_iterator_latency_ps: 80000000
                  iterator_stats {
                    key: 123,
                    value: {
                      id: 123
                      start_time_ps: 100000000
                      duration_ps: 100000000
                      self_time_ps: 20000000
                      is_blocking: true
                      num_calls: 1
                    }
                  }
                  iterator_stats {
                    key: 456,
                    value: {
                      id: 456
                      start_time_ps: 100000000
                      duration_ps: 80000000
                      self_time_ps: 80000000
                      is_blocking: true
                      num_calls: 1
                    }
                  }
                }
                stats {
                  bottleneck_iterator_id: 123
                  bottleneck_iterator_latency_ps: 30000000
                  iterator_stats {
                    key: 123,
                    value: {
                      id: 123
                      start_time_ps: 0
                      duration_ps: 30000000
                      self_time_ps: 30000000
                      is_blocking: true
                      num_calls: 1
                    }
                  }
                }
              }
            }
          }
        }
        summary: "No tf.data activity captured in your profile. If your job uses tf.data, try to capture a longer profile."
      )pb"));
}

// Test with the following example dataset:
// dataset = tf.data.Dataset.range(8)
// dataset = dataset.map(lambda x: x + 1)
// dataset = dataset.batch(2)
// for _ in dataset:
//   pass
TEST(XPlaneToTfDataStatsTest, MapAndBatch) {
  constexpr int64_t kMapAndBatchIteratorId = 123;
  constexpr int64_t kRangeIteratorId = 456;
  constexpr int64_t kElementId = 100;

  XPlane host_plane;
  XPlaneBuilder host_plane_builder(&host_plane);
  host_plane_builder.ReserveLines(2);

  XLineBuilder consumer_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &consumer_thread, "Iterator::MapAndBatch",
               0, 100000000, {{StatType::kStepId, kMapAndBatchIteratorId}});
  CreateXEvent(&host_plane_builder, &consumer_thread,
               HostEventType::kMapAndBatchConsume, 80000000, 20000000,
               {{StatType::kElementId, kElementId}});

  XLineBuilder producer_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kMapAndBatchProduce, 0, 30000000,
               {{StatType::kElementId, kElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::MapAndBatch::Range", 0, 30000000,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kMapAndBatchIteratorId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               HostEventType::kMapAndBatchProduce, 40000000, 30000000,
               {{StatType::kElementId, kElementId}});
  CreateXEvent(&host_plane_builder, &producer_thread,
               "Iterator::MapAndBatch::Range", 40000000, 30000000,
               {{StatType::kStepId, kRangeIteratorId},
                {StatType::kParentId, kMapAndBatchIteratorId}});

  CombinedTfDataStats combined_tf_data_stats;
  CombinedTfDataStatsBuilder builder(&combined_tf_data_stats);
  builder.Add("host1", &host_plane);
  builder.Finalize();
  EXPECT_THAT(
      combined_tf_data_stats, EqualsProto(R"pb(
        bottleneck_analysis: {
          host: "host1"
          input_pipeline: "Host:0"
          max_latency_ps: 100000000
          iterator_name: "Range"
          iterator_long_name: "Iterator::MapAndBatch::Range"
          iterator_latency_ps: 60000000
          suggestion: "See <a href=\"https://www.tensorflow.org/guide/data_performance_analysis\" target=\"_blank\">this</a> for suggestions."
        }
        tf_data_stats: {
          key: "host1"
          value: {
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
                avg_latency_ps: 100000000
                min_latency_ps: 100000000
                max_latency_ps: 100000000
                num_slow_calls: 1
                stats {
                  bottleneck_iterator_id: 456
                  bottleneck_iterator_latency_ps: 60000000
                  iterator_stats {
                    key: 123,
                    value: {
                      id: 123
                      start_time_ps: 0
                      duration_ps: 100000000
                      self_time_ps: 40000000
                      is_blocking: true
                      num_calls: 1
                    }
                  }
                  iterator_stats {
                    key: 456,
                    value: {
                      id: 456
                      start_time_ps: 0
                      duration_ps: 60000000
                      self_time_ps: 60000000
                      is_blocking: true
                      num_calls: 2
                    }
                  }
                }
              }
            }
          }
        }
        is_input_bound: true
        summary: "Your profile has a tf.data input pipeline slower than 50 us. For each slow input pipeline, below shows a bottleneck in the input pipeline and a suggestion on how to fix it."
      )pb"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
