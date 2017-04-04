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

#include "tensorflow/contrib/xla_tf_graph/xla_tf_graph_util.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"

namespace tensorflow {
namespace xla_tf_graph {

namespace {

constexpr const char* const GRAPH_NAME = "xla_tf_graph_util";

void SetupXlaCpuClient(std::unique_ptr<FunctionLibraryDefinition>* flib_def,
                       std::unique_ptr<FunctionLibraryRuntime>* flr,
                       std::unique_ptr<XlaCompiler>* compiler) {
  xla::Client* client = xla::ClientLibrary::LocalClientOrDie();
  XlaOpRegistry::RegisterCompilationKernels();

  FunctionDefLibrary flib;
  flib_def->reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));

  // Setup compiler options
  XlaCompiler::Options options;
  options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);
  options.client = client;
  compiler->reset(new XlaCompiler(options));

  flr->reset(NewFunctionLibraryRuntime(
      compiler->get()->device_mgr(), /*env=*/nullptr, compiler->get()->device(),
      TF_GRAPH_DEF_VERSION, flib_def->get(), OptimizerOptions(),
      /*custom_kernel_creator=*/nullptr));
}

}  // namespace

xla::StatusOr<std::unique_ptr<xla::SessionModule>>
ConvertTfGraphToXlaSessionModule(const std::vector<XlaCompiler::Argument>& args,
                                 std::unique_ptr<Graph> graph) {
  CHECK(graph);

  std::unique_ptr<FunctionLibraryDefinition> flib_def;
  std::unique_ptr<FunctionLibraryRuntime> flr;
  std::unique_ptr<XlaCompiler> compiler;

  SetupXlaCpuClient(&flib_def, &flr, &compiler);

  // Compile graph and build computation
  XlaCompiler::CompilationResult result;
  TF_CHECK_OK(compiler->CompileGraph(GRAPH_NAME, std::move(graph), flr.get(),
                                     args, &result));

  return result.computation.Snapshot();
}

}  // namespace xla_tf_graph
}  // namespace tensorflow
