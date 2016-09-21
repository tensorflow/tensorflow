/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STATS_PUBLISHER_INTERFACE_H_
#define THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STATS_PUBLISHER_INTERFACE_H_

#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// StatsPublisherInterface describes objects that publish information exported
// by Sessions.
// Implementations must be thread-safe.
class StatsPublisherInterface {
 public:
  // PublishStatsProto publishes step_stats.
  // When PublishStatsProto is called multiple times, only the step_stats
  // corresponding to the latest call will be published.
  virtual void PublishStatsProto(const StepStats& step_stats) = 0;

  // PublishGraphProto publishes the graph_defs corresponding to each partition
  // in the session.
  // When PublishGraphProto is called multiple times, only the graph_defs
  // corresponding to the latest call will be published.
  virtual void PublishGraphProto(const vector<const GraphDef*>& graph_defs) = 0;

  // TODO(suharshs): Publish timeline.

  virtual ~StatsPublisherInterface() {}
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_STATS_PUBLISHER_INTERFACE_H_
