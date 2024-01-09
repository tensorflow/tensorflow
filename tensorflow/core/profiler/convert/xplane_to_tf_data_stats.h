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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TF_DATA_STATS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TF_DATA_STATS_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/tf_data_stats.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

TF_CONST_INIT extern const int64_t kSlowCallThresholdPs;

enum class BottleneckType {
  kSlowSource,
  kSlowDataService,
  kSlowRemoteSource,
  kSlowTransformationWithParallelVersion,
  kSlowTransformationWithoutParallelVersion,
  kOther,
};

BottleneckType GetBottleneckType(absl::string_view bottleneck_iterator_name);

class CombinedTfDataStatsBuilder {
 public:
  explicit CombinedTfDataStatsBuilder(
      CombinedTfDataStats* combined_tf_data_stats,
      bool generate_suggestion = true)
      : combined_tf_data_stats_(combined_tf_data_stats),
        generate_suggestion_(generate_suggestion) {}

  void Add(absl::string_view host_name, XPlane* host_plane);

  // Finalizes by populating TfDataBottleneckAnalysis.
  void Finalize();

 private:
  CombinedTfDataStats* combined_tf_data_stats_;
  bool generate_suggestion_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XPLANE_TO_TF_DATA_STATS_H_
