/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_AOT_COMPILE_H_
#define TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_AOT_COMPILE_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/compiler.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime

namespace tensorflow::tfrt_stub {
struct AotOptions {
  AotOptions();
  std::unordered_set<std::string> tags = {};
  std::shared_ptr<GraphExecutionOptions> graph_execution_options;
  // TODO(b/296466237): support compiling for multiple signature functions.
  // The signature name to be AOT compiled.
  std::string signature_name;
};

struct AotResult {
  using ExecutableMap =
      absl::flat_hash_map<DeviceCompilationClusterSignature, std::string,
                          DeviceCompilationClusterSignature::Hash>;
  std::variant<tfrt::BefBuffer, mlrt::bc::Buffer> buffer;
  // TODO(b/296466237): Investigate whether the whole FunctionDefLibrary should
  // be put here.
  // XLA cluster functions corresponding to `XlaLaunch` op, generated during
  // bridge.
  std::vector<FunctionDef> xla_functions;
};

// AOT compiles saved_model in input_model_dir and returns AotResult, otherwise
// returns error.
StatusOr<AotResult> AotCompileSavedModel(absl::string_view input_model_dir,
                                         AotOptions aot_options = {});

// TODO(b/296466237): Add unit test.
// Runs bridge and compiles the generated XLA functions corresponding to the
// signature function with name `siganture_name` in MetaGraphDef.
// `input_shapes` maps input signature node name to its tensor shape, and is
// used to make up for the missing input shape information in the graph if any
// so that shape inference pass in bridge can proceed correctly. Returns
// AotResult::ExecutableMap as compilation result, which maps function
// signatures to serialized executables.
StatusOr<AotResult::ExecutableMap> AotCompileXlaFunctionsInMetaGraphDef(
    const MetaGraphDef& meta_graph_def, const std::string& signature_name,
    const absl::flat_hash_map<std::string, tensorflow::TensorShapeProto>&
        input_shapes,
    const tensorflow::FunctionDefLibrary& fdef_lib,
    const tensorflow::SessionOptions& session_options,
    const mlir::DialectRegistry& registry, const AotOptions& aot_options,
    absl::string_view input_model_dir, ModelRuntimeContext& model_context);

// TODO(b/296466237): make this function general for all devices.
// AOT compiles `function` into PjRtExecutable. It is the counterpart of the JIT
// version `CompileToPjRtLoadedExecutable`. `compilation_result` contains the
// generated XLA computation.
StatusOr<std::unique_ptr<xla::PjRtExecutable>> AotCompileToGpuPjRtExecutable(
    const FunctionLibraryDefinition* flib_def, const NameAttrList& function,
    int graph_def_version, const std::vector<XlaCompiler::Argument>& args,
    bool has_ref_vars, bool may_alias_resource_update,
    const stream_executor::GpuTargetConfigProto& gpu_target_config,
    XlaCompiler::CompilationResult** compilation_result);

// Returns serialized PJRT loaded GPU executable. This function requires GPU
// device to be present during compilation.
StatusOr<std::string> AotCompileToGpuPjRtLoadedExecutableWithDevice(
    const FunctionLibraryDefinition* flib_def, const NameAttrList& function,
    int graph_def_version, const std::vector<XlaCompiler::Argument>& args,
    bool has_ref_vars, bool may_alias_resource_update,
    XlaCompiler::CompilationResult** compilation_result);
}  // namespace tensorflow::tfrt_stub

#endif  // TENSORFLOW_CORE_TFRT_SAVED_MODEL_SAVED_MODEL_AOT_COMPILE_H_
