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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_PRINT_MODEL_ANALYSIS_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_PRINT_MODEL_ANALYSIS_H_

#include <string>

namespace tensorflow {
namespace tfprof {
struct Options;

// **********************
// APIs in this file are only for swig.
// Talk to xpan@ if you want to call it directly!
// *********************

// Multi-step Profiler.
//
bool NewProfiler(const std::string* graph, const std::string* op_log);

void DeleteProfiler();

double AddStep(int64_t step, const std::string* graph,
               const std::string* run_meta, const std::string* op_log);

// Write the profiler's profile to a proto buffer.
void WriteProfile(const std::string* filename);

// Load the profile to profiler from a proto buffer file.
void ProfilerFromFile(const std::string* filename);

// Returns a binary string that represents the serialized ProfileProto.
std::string SerializeToString();

std::string Profile(const std::string* command, const std::string* options);

// Single-step Profiler.
//
// Interface defined for Python API swig. Calls the tfprof core API.
// 'graph', 'run_meta', 'op_log' are serialized GraphDef, RunMetadata,
// OpLogProto strings, respectively.
// 'graph', 'command' and 'options' are required. Others can be nullptr
// if not available.
std::string PrintModelAnalysis(const std::string* graph,
                               const std::string* run_meta,
                               const std::string* op_log,
                               const std::string* command,
                               const std::string* options);

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_PRINT_MODEL_ANALYSIS_H_
