/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_INTERNAL_CHECKER_RUNNER_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_INTERNAL_CHECKER_RUNNER_H_

#include "tensorflow/tools/tfprof/internal/tfprof_utils.h"

namespace tensorflow {
namespace tfprof {

class TFStats;

std::map<string, std::vector<string>> RunInternalCheckers(const TFStats* stats);

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_ADVISOR_INTERNAL_CHECKER_RUNNER_H_
