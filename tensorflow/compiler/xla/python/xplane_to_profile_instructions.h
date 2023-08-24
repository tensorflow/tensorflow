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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_XPLANE_TO_PROFILE_INSTRUCTIONS_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_XPLANE_TO_PROFILE_INSTRUCTIONS_H_

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {

// Separator for fingerprint and hlo_name in the cost name of
// ProfiledInstructionsProto.
extern const char kCostNameSep[];

// Latency info for a single HLO instruction.
struct HloLatencyInfo {
  std::vector<double> durations;
};

// Convert XSpace to ProfiledInstructionsProto. This function will aggregate
// all the xplane.pb info into ProfiledInstructionsProto.
Status ConvertXplaneToProfiledInstructionsProto(
    const std::string& logdir, tensorflow::profiler::ProfiledInstructionsProto*
                                   profiled_instructions_proto);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_XPLANE_TO_PROFILE_INSTRUCTIONS_H_
