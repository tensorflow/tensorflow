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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_TFPROF_TOOLS_TFPROF_INTERNAL_PRINT_MODEL_ANALYSIS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_TFPROF_TOOLS_TFPROF_INTERNAL_PRINT_MODEL_ANALYSIS_H_

#include <string>

#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/tfprof_log.pb.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/tfprof_output.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {

// ***This API is only for swig.***
//
// Interface defined for Python API swig. Calls the tfprof core API.
// 'graph', 'run_meta', 'op_log' are serialized GraphDef, RunMetadata,
// OpLog strings, respectively.
// 'graph', 'command' and 'options' are required. Others can be nullptr
// if not available.
string PrintModelAnalysis(const string* graph, const string* run_meta,
                          const string* op_log, const string* command,
                          const Options* options);

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_TFPROF_TOOLS_TFPROF_INTERNAL_PRINT_MODEL_ANALYSIS_H_
