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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_JIT_COMPILED_CPU_FUNCTION_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_JIT_COMPILED_CPU_FUNCTION_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Represents the result of JIT compilation by XLA down to a function. This
// class holds the state necessary to create XlaCompiledCpuFunction instances,
// which are used to actually invoke the compiled computation.
//
// XlaJitCompiledCpuFunction must outlive the XlaCompiledCpuFunctions that are
// created from it. It holds state shared by all of the functions, including the
// JIT-compiled function itself, along with buffer sizes and other metadata
// necessary for execution.
class XlaJitCompiledCpuFunction {
 public:
  // Compile a tensorflow::GraphDef into an XlaJitCompiledCpuFunction. The given
  // `config` specifies the portion of the graph to compile, via feeds and
  // fetches. Each feed is a positional input argument for the compiled
  // function, while each fetch is a positional output argument.
  static xla::StatusOr<std::unique_ptr<XlaJitCompiledCpuFunction>> Compile(
      const GraphDef& graph_def, const tf2xla::Config& config,
      const xla::ExecutableBuildOptions& build_options);

  XlaJitCompiledCpuFunction(const XlaJitCompiledCpuFunction&) = delete;
  XlaJitCompiledCpuFunction& operator=(const XlaJitCompiledCpuFunction&) =
      delete;

  // Returns static data used to create an XlaCompiledCpuFunction instance,
  // which represents the JIT-compiled function. The static data is unchanging
  // across each instance.
  const XlaCompiledCpuFunction::StaticData& StaticData() const {
    return static_data_;
  }

 private:
  XlaJitCompiledCpuFunction() {}

  // The executable holds the underlying function.
  std::unique_ptr<xla::LocalExecutable> executable_;

  // The static data is backed by the rest of the state in this class.
  XlaCompiledCpuFunction::StaticData static_data_;

  // The backing array for buffer infos.
  std::vector<xla::cpu_function_runtime::BufferInfo> buffer_infos_;

  // The backing array for the arg index table.
  std::vector<int32> arg_index_table_;

  // The backing arrays of arg and result names. We hold the actual strings in
  // nonempty_*_names_, and hold arrays of pointers in *_names_ for the static
  // data to refer to.
  std::vector<string> nonempty_arg_names_;
  std::vector<string> nonempty_variable_names_;
  std::vector<string> nonempty_result_names_;
  std::vector<const char*> arg_names_;
  std::vector<const char*> variable_names_;
  std::vector<const char*> result_names_;

  // The backing data for the program shape. The proto form of program shape is
  // used because the program shape is serialized and embedded in the object
  // file.
  std::unique_ptr<const xla::ProgramShapeProto> program_shape_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_JIT_COMPILED_CPU_FUNCTION_H_
