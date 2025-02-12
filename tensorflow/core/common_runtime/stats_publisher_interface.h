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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_STATS_PUBLISHER_INTERFACE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_STATS_PUBLISHER_INTERFACE_H_

#include <functional>
#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/profile_handler.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class StatsPublisherInterface;

typedef std::function<std::unique_ptr<StatsPublisherInterface>(
    const std::string&, const BuildGraphOptions&, const SessionOptions&)>
    StatsPublisherFactory;

// StatsPublisherInterface describes objects that publish information exported
// by Sessions.
// NOTE: This interface is experimental and subject to change.
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
  virtual void PublishGraphProto(
      const std::vector<const GraphDef*>& graph_defs) = 0;
  virtual void PublishGraphProto(std::vector<GraphDef> graph_defs) = 0;
  virtual void PublishGraphProto(
      std::vector<core::RefCountPtr<FunctionRecord>>&& function_records) = 0;

  // Returns a profile handler for the given step based on the execution_count
  // and RunOptions.
  //
  // This method may return a null pointer, if no handler was created.
  virtual std::unique_ptr<ProfileHandler> GetProfileHandler(
      uint64 step, int64_t execution_count, const RunOptions& ropts) = 0;

  virtual ~StatsPublisherInterface() {}

  static void RegisterStatsPublisher(StatsPublisherFactory factory_fn);

  static StatsPublisherFactory GetStatsPublisherFactory();

 private:
  static StatsPublisherFactory** GetStatsPublisherFactoryPtr() {
    static StatsPublisherFactory* stats_publisher_factory = nullptr;
    return &stats_publisher_factory;
  }
};

std::unique_ptr<StatsPublisherInterface> CreateNoOpStatsPublisher(
    const string& session, const BuildGraphOptions& bopts,
    const SessionOptions& sopts);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_STATS_PUBLISHER_INTERFACE_H_
