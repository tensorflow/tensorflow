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

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/refcount.h"

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

  void PublishGraphProto(std::vector<GraphDef> graph_defs) override {}

  void PublishGraphProto(std::vector<core::RefCountPtr<FunctionRecord>>&&
                             function_records) override {}

  std::unique_ptr<ProfileHandler> GetProfileHandler(
      uint64 step, int64_t execution_count, const RunOptions& ropts) override {
    return nullptr;
  }

  ~NoOpStatsPublisher() override = default;
};

}  // namespace

void StatsPublisherInterface::RegisterStatsPublisher(
    StatsPublisherFactory factory_fn) {
  StatsPublisherFactory** factory_ptr = GetStatsPublisherFactoryPtr();
  if (*factory_ptr == nullptr) {
    *factory_ptr = new StatsPublisherFactory();
  } else {
    LOG(WARNING)
        << "More than one StatsPublisherFactory functions are registered. Only "
           "the last registered one will be effective.";
  }
  **factory_ptr = factory_fn;
}

StatsPublisherFactory StatsPublisherInterface::GetStatsPublisherFactory() {
  const auto* factory_ptr = GetStatsPublisherFactoryPtr();
  if (*factory_ptr == nullptr) {
    return CreateNoOpStatsPublisher;
  }
  return **factory_ptr;
}

std::unique_ptr<StatsPublisherInterface> CreateNoOpStatsPublisher(
    const string& session, const BuildGraphOptions& bopts,
    const SessionOptions& sopts) {
  return std::unique_ptr<StatsPublisherInterface>(new NoOpStatsPublisher);
}

}  // namespace tensorflow
