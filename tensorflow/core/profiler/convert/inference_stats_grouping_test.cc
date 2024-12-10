/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/convert/inference_stats_grouping.h"

#include <gmock/gmock.h>
#include "xla/tests/test_utils.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"

namespace tensorflow::profiler {
namespace {

using ::testing::EqualsProto;
using ::xla::ParseTextProto;

TEST(InferenceStatsGroupingTest, TestWithModelId) {
  // An inference stats with two hosts, two models.
  InferenceStats inference_stats = ParseTextProto<InferenceStats>(R"pb(
                                     inference_stats_per_host {
                                       key: 0
                                       value {
                                         request_details {
                                           start_time_ps: 1000
                                           end_time_ps: 2000
                                           model_id_index: 0
                                           request_id: 0
                                           device_time_ps: 100
                                         }
                                         request_details {
                                           start_time_ps: 2000
                                           end_time_ps: 3000
                                           model_id_index: 1
                                           request_id: 1
                                           device_time_ps: 100
                                         }
                                       }
                                     }
                                     inference_stats_per_host {
                                       key: 1
                                       value {
                                         request_details {
                                           start_time_ps: 3000
                                           end_time_ps: 4000
                                           model_id_index: 0
                                           request_id: 2
                                           device_time_ps: 100
                                         }
                                         request_details {
                                           start_time_ps: 4000
                                           end_time_ps: 5000
                                           model_id_index: 1
                                           request_id: 3
                                           device_time_ps: 100
                                         }
                                       }
                                     }
                                     model_id_db {
                                       ids: "Model-A:1"
                                       ids: "Model-B:1"
                                       id_to_index { key: "Model-A:1" value: 0 }
                                       id_to_index { key: "Model-B:1" value: 1 }
                                     }
                                   )pb")
                                       .value();

  RegroupInferenceStatsByModel(&inference_stats);

  // Verifies that requests with the same model ID are grouped together.
  EXPECT_THAT(inference_stats, EqualsProto(R"pb(
                model_id_db {
                  ids: "Model-A:1"
                  ids: "Model-B:1"
                  id_to_index { key: "Model-A:1" value: 0 }
                  id_to_index { key: "Model-B:1" value: 1 }
                }
                inference_stats_per_model {
                  key: 0
                  value {
                    request_details {
                      start_time_ps: 1000
                      end_time_ps: 2000
                      model_id_index: 0
                      request_id: 0
                      device_time_ps: 100
                    }
                    request_details {
                      start_time_ps: 3000
                      end_time_ps: 4000
                      model_id_index: 0
                      request_id: 2
                      device_time_ps: 100
                    }
                    aggregated_request_detail {
                      request_id: -1
                      start_time_ps: 0
                      end_time_ps: 1000
                      write_to_device_time_ps: 0
                      read_from_device_time_ps: 0
                      device_time_ps: 100
                      batching_request_delay_ps: 0
                      batching_request_size: 0
                      host_preprocessing_ps: 0
                      host_batch_formation_ps: 0
                      host_runtime_ps: 0
                      host_postprocessing_ps: 0
                      idle_time_ps: 0
                    }
                    aggregated_batch_detail {}
                    request_throughput: 666666666.66666663
                    request_average_latency_us: 0.001
                    batch_throughput: 0
                    batch_average_latency_us: 0
                  }
                }
                inference_stats_per_model {
                  key: 1
                  value {
                    request_details {
                      start_time_ps: 2000
                      end_time_ps: 3000
                      model_id_index: 1
                      request_id: 1
                      device_time_ps: 100
                    }
                    request_details {
                      start_time_ps: 4000
                      end_time_ps: 5000
                      model_id_index: 1
                      request_id: 3
                      device_time_ps: 100
                    }
                    aggregated_request_detail {
                      request_id: -1
                      start_time_ps: 0
                      end_time_ps: 1000
                      write_to_device_time_ps: 0
                      read_from_device_time_ps: 0
                      device_time_ps: 100
                      batching_request_delay_ps: 0
                      batching_request_size: 0
                      host_preprocessing_ps: 0
                      host_batch_formation_ps: 0
                      host_runtime_ps: 0
                      host_postprocessing_ps: 0
                      idle_time_ps: 0
                    }
                    aggregated_batch_detail {}
                    request_throughput: 666666666.66666663
                    request_average_latency_us: 0.001
                    batch_throughput: 0
                    batch_average_latency_us: 0
                  }
                })pb"));
}

TEST(InferenceStatsGroupingTest, TestTensorPatternPercentile) {
  // Generates an inference stats for test, 6 requests have tensor events owned
  // by REQUEST, 2 requests have tensor events owned by BATCH.
  InferenceStats inference_stats =
      ParseTextProto<InferenceStats>(R"pb(
        inference_stats_per_host {
          key: 0
          value {
            request_details {
              start_time_ps: 1000
              end_time_ps: 2000
              request_id: 0
              tensor_event_details {
                tensor_pattern_index: 0
                owner: REQUEST
                linearize_delinearize_time_ps: 600000
              }
            }
            request_details {
              start_time_ps: 2000
              end_time_ps: 3000
              request_id: 1
              tensor_event_details {
                tensor_pattern_index: 0
                owner: REQUEST
                linearize_delinearize_time_ps: 500000
              }
            }
            request_details {
              start_time_ps: 1000
              end_time_ps: 2000
              request_id: 2
              tensor_event_details {
                tensor_pattern_index: 0
                owner: REQUEST
                linearize_delinearize_time_ps: 400000
              }
            }
            request_details {
              start_time_ps: 2000
              end_time_ps: 3000
              request_id: 3
              tensor_event_details {
                tensor_pattern_index: 0
                owner: REQUEST
                linearize_delinearize_time_ps: 300000
              }
            }
            request_details {
              start_time_ps: 1000
              end_time_ps: 2000
              request_id: 4
              tensor_event_details {
                tensor_pattern_index: 0
                owner: REQUEST
                linearize_delinearize_time_ps: 200000
              }
            }
            request_details {
              start_time_ps: 2000
              end_time_ps: 3000
              request_id: 5
              tensor_event_details {
                tensor_pattern_index: 0
                owner: REQUEST
                linearize_delinearize_time_ps: 100000
              }
            }
            request_details {
              start_time_ps: 2000
              end_time_ps: 3000
              request_id: 6
              tensor_event_details {
                tensor_pattern_index: 0
                owner: BATCH
                linearize_delinearize_time_ps: 700000
              }
            }
            request_details {
              start_time_ps: 2000
              end_time_ps: 3000
              request_id: 7
              tensor_event_details {
                tensor_pattern_index: 0
                owner: BATCH
                linearize_delinearize_time_ps: 800000
              }
            }
          }
        }
      )pb")
          .value();

  RegroupInferenceStatsByModel(&inference_stats);

  // Count equals to 6 because request tensor events owned by BATCH are ignored.
  // Percentile selector selects linearize and delinearize time at 50.0, 75.0,
  // 90.0, 95.0, 99.0, 99.9 percentiles.
  EXPECT_THAT(inference_stats.inference_stats_per_model()
                  .at(0)
                  .tensor_transfer_aggregated_result(),
              EqualsProto(R"pb(
                tensor_pattern_results {
                  tensor_pattern_index: 0
                  count: 6
                  linearize_delinearize_percentile_time {
                    percentile: 50
                    time_ps: 400000
                  }
                  linearize_delinearize_percentile_time {
                    percentile: 75
                    time_ps: 500000
                  }
                  linearize_delinearize_percentile_time {
                    percentile: 90
                    time_ps: 600000
                  }
                  linearize_delinearize_percentile_time {
                    percentile: 95
                    time_ps: 600000
                  }
                  linearize_delinearize_percentile_time {
                    percentile: 99
                    time_ps: 600000
                  }
                  linearize_delinearize_percentile_time {
                    percentile: 99.9
                    time_ps: 600000
                  }
                }
              )pb"));
}

TEST(InferenceStatsGroupingTest, TestWithoutModelId) {
  // An inference stats with two hosts, no model ID data.
  InferenceStats inference_stats = ParseTextProto<InferenceStats>(R"pb(
                                     inference_stats_per_host {
                                       key: 0
                                       value {
                                         request_details {
                                           start_time_ps: 1000
                                           end_time_ps: 2000
                                           request_id: 0
                                           related_batch_ids: 0
                                           host_runtime_ps: 100
                                         }
                                         request_details {
                                           start_time_ps: 2000
                                           end_time_ps: 4000
                                           request_id: 1
                                           related_batch_ids: 0
                                           host_runtime_ps: 100
                                         }
                                         batch_details {
                                           batch_id: 0
                                           related_request_ids: 0
                                           related_request_ids: 1
                                           start_time_ps: 1000
                                           end_time_ps: 2000
                                           batch_size_after_padding: 128
                                         }
                                       }
                                     }
                                     inference_stats_per_host {
                                       key: 1
                                       value {
                                         request_details {
                                           start_time_ps: 3000
                                           end_time_ps: 6000
                                           request_id: 2
                                           related_batch_ids: 1
                                           host_runtime_ps: 100
                                         }
                                         request_details {
                                           start_time_ps: 4000
                                           end_time_ps: 8000
                                           request_id: 3
                                           related_batch_ids: 1
                                           host_runtime_ps: 100
                                         }
                                         batch_details {
                                           batch_id: 1
                                           related_request_ids: 2
                                           related_request_ids: 3
                                           start_time_ps: 3000
                                           end_time_ps: 4000
                                           batch_size_after_padding: 256
                                         }
                                       }
                                     }
                                   )pb")
                                       .value();

  RegroupInferenceStatsByModel(&inference_stats);

  // Verifies that all requests are grouped into a single model, and a "ALL"
  // model ID is added.
  EXPECT_THAT(inference_stats, EqualsProto(R"pb(
                model_id_db {
                  ids: "ALL"
                  id_to_index { key: "ALL" value: 0 }
                }
                inference_stats_per_model {
                  key: 0
                  value {
                    request_details {
                      start_time_ps: 1000
                      end_time_ps: 2000
                      request_id: 0
                      related_batch_ids: 0
                      host_runtime_ps: 100
                    }
                    request_details {
                      start_time_ps: 2000
                      end_time_ps: 4000
                      request_id: 1
                      related_batch_ids: 0
                      host_runtime_ps: 100
                    }
                    request_details {
                      start_time_ps: 3000
                      end_time_ps: 6000
                      request_id: 2
                      related_batch_ids: 1
                      host_runtime_ps: 100
                    }
                    request_details {
                      start_time_ps: 4000
                      end_time_ps: 8000
                      request_id: 3
                      related_batch_ids: 1
                      host_runtime_ps: 100
                    }
                    batch_details {
                      batch_id: 0
                      related_request_ids: 0
                      related_request_ids: 1
                      start_time_ps: 1000
                      end_time_ps: 2000
                      batch_size_after_padding: 128
                    }
                    batch_details {
                      batch_id: 1
                      related_request_ids: 2
                      related_request_ids: 3
                      start_time_ps: 3000
                      end_time_ps: 4000
                      batch_size_after_padding: 256
                    }
                    aggregated_request_detail {
                      request_id: -1
                      start_time_ps: 0
                      end_time_ps: 2500
                      write_to_device_time_ps: 0
                      read_from_device_time_ps: 0
                      device_time_ps: 0
                      batching_request_delay_ps: 0
                      batching_request_size: 0
                      host_preprocessing_ps: 0
                      host_batch_formation_ps: 0
                      host_runtime_ps: 100
                      host_postprocessing_ps: 0
                      idle_time_ps: 0
                    }
                    aggregated_batch_detail {
                      batch_id: -1
                      start_time_ps: 0
                      end_time_ps: 1000
                      batch_delay_ps: 0
                      padding_amount: 0
                      batch_size_after_padding: 192
                    }
                    per_batch_size_aggregated_result {
                      batch_size: 128
                      aggregated_request_result {
                        start_time_ps: 0
                        end_time_ps: 1500
                        write_to_device_time_ps: 0
                        read_from_device_time_ps: 0
                        device_time_ps: 0
                        request_id: -1
                        batching_request_delay_ps: 0
                        batching_request_size: 0
                        host_preprocessing_ps: 0
                        host_batch_formation_ps: 0
                        host_runtime_ps: 100
                        host_postprocessing_ps: 0
                        idle_time_ps: 0
                      }
                      aggregated_batch_result {
                        batch_id: -1
                        start_time_ps: 0
                        end_time_ps: 1000
                        batch_delay_ps: 0
                        padding_amount: 0
                        batch_size_after_padding: 128
                      }
                      request_throughput: 285714285.71428573
                      batch_throughput: 333333333.33333331
                    }
                    per_batch_size_aggregated_result {
                      batch_size: 256
                      aggregated_request_result {
                        start_time_ps: 0
                        end_time_ps: 3500
                        write_to_device_time_ps: 0
                        read_from_device_time_ps: 0
                        device_time_ps: 0
                        request_id: -1
                        batching_request_delay_ps: 0
                        batching_request_size: 0
                        host_preprocessing_ps: 0
                        host_batch_formation_ps: 0
                        host_runtime_ps: 100
                        host_postprocessing_ps: 0
                        idle_time_ps: 0
                      }
                      aggregated_batch_result {
                        batch_id: -1
                        start_time_ps: 0
                        end_time_ps: 1000
                        batch_delay_ps: 0
                        padding_amount: 0
                        batch_size_after_padding: 256
                      }
                      request_throughput: 285714285.71428573
                      batch_throughput: 333333333.33333331
                    }
                    request_throughput: 571428571.42857146
                    request_average_latency_us: 0.0025
                    batch_throughput: 666666666.66666663
                    batch_average_latency_us: 0.001
                  }
                })pb"));
}

}  // namespace
}  // namespace tensorflow::profiler
