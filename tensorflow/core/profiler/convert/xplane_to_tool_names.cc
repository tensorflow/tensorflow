/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_tool_names.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/repository.h"
#include "tensorflow/core/profiler/convert/xplane_to_hlo.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"

namespace tensorflow {
namespace profiler {

StatusOr<std::string> GetAvailableToolNames(
    const SessionSnapshot& session_snapshot) {
  std::vector<std::string> tools;
  if (session_snapshot.XSpaceSize() != 0) {
    tools.reserve(11);
    tools.push_back("trace_viewer");
    tools.push_back("overview_page");
    tools.push_back("input_pipeline_analyzer");
    tools.push_back("tensorflow_stats");
    tools.push_back("memory_profile");
    tools.push_back("pod_viewer");
    tools.push_back("tf_data_bottleneck_analysis");
    tools.push_back("op_profile");

    TF_ASSIGN_OR_RETURN(std::unique_ptr<XSpace> xspace,
                        session_snapshot.GetXSpace(0));

    if (!FindPlanesWithPrefix(*xspace, kGpuPlanePrefix).empty()) {
      tools.push_back("kernel_stats");
    }

    TF_ASSIGN_OR_RETURN(bool has_hlo,
                        ConvertMultiXSpaceToHloProto(session_snapshot));
    if (has_hlo) {
      tools.push_back("memory_viewer");
      tools.push_back("graph_viewer");
    }
  }

  return absl::StrJoin(tools, ",");
}

}  // namespace profiler
}  // namespace tensorflow
