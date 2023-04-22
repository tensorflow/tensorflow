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

#include "tensorflow/core/common_runtime/stats_publisher_interface.h"

#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {
namespace {

// NoOpStatsPublisher provides an dummy/no-op implementation of
// StatsPublisherInterface.
class NoOpStatsPublisher : public StatsPublisherInterface {
 public:
  NoOpStatsPublisher() = default;

  void PublishStatsProto(const StepStats& step_stats) override {}

  void PublishGraphProto(
      const std::vector<const GraphDef*>& graph_defs) override {}

  std::unique_ptr<ProfileHandler> GetProfileHandler(
      uint64 step, int64_t execution_count, const RunOptions& ropts) override {
    return nullptr;
  }

  ~NoOpStatsPublisher() override = default;
};

}  // namespace

std::unique_ptr<StatsPublisherInterface> CreateNoOpStatsPublisher(
    const string& session, const BuildGraphOptions& bopts,
    const SessionOptions& sopts) {
  return std::unique_ptr<StatsPublisherInterface>(new NoOpStatsPublisher);
}

}  // namespace tensorflow
