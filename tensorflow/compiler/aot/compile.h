/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_AOT_COMPILE_H_
#define TENSORFLOW_COMPILER_AOT_COMPILE_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/aot/flags.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tfcompile {

// CompileResult describes the output of CompileGraph, where the object file
// data and meta-information is available in aot.
struct CompileResult {
  // Contains object file and meta-info.
  std::unique_ptr<xla::cpu::CpuAotCompilationResultLegacy> aot;
  xla::ProgramShapeProto program_shape;  // Static shape of args and results.
  string entry_point;                    // Name of generated function.
  int pointer_size = 0;                  // Size of a pointer in bytes.
};

// CompileGraph compiles the graph_def into an object file containing a function
// that performs the graph operations.
//
// The XLA compilation options are specified in the flags.
absl::Status CompileGraph(GraphDef graph_def, const tf2xla::Config& config,
                          const MainFlags& flags,
                          CompileResult* compile_result);

// The full compilation method, for reuse in a library setting.
absl::Status Main(const MainFlags& flags);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_COMPILE_H_
