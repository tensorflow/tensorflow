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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_PRINT_MODEL_ANALYSIS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_PRINT_MODEL_ANALYSIS_H_

#include <string>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace tfprof {
struct Options;

// **********************
// APIs in this file are only for swig.
// Talk to xpan@ if you want to call it directly!
// *********************

// Multi-step Profiler.
//
bool NewProfiler(const string* graph, const string* op_log);

void DeleteProfiler();

void AddStep(int64 step, const string* run_meta, const string* op_log);

string Profile(const string* command, const string* options);

// Single-step Profiler.
//
// Interface defined for Python API swig. Calls the tfprof core API.
// 'graph', 'run_meta', 'op_log' are serialized GraphDef, RunMetadata,
// OpLogProto strings, respectively.
// 'graph', 'command' and 'options' are required. Others can be nullptr
// if not available.
string PrintModelAnalysis(const string* graph, const string* run_meta,
                          const string* op_log, const string* command,
                          const string* options);

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_PROFILER_INTERNAL_PRINT_MODEL_ANALYSIS_H_
