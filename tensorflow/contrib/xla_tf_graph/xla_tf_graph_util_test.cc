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
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace xla_tf_graph {

static std::unique_ptr<Graph> BuildAddGraph() {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  // See tf2xla/kernels/binary_ops.cc
  auto c = ops::Add(scope.WithOpName("C"), a, b);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_CHECK_OK(scope.ToGraph(graph.get()));
  return graph;
}

static std::vector<XlaCompiler::Argument> BuildAddGraphArguments() {
  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  // Difference of dimension will add extra broadcast_dimensions.
  // broadcast_dimension generates an additional HloInstruction
  // in user_computation.cc
  args[0].shape = xla::ShapeUtil::MakeShape(xla::S32, {2, 2});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(xla::S32, {2});
  return args;
}

// CAVEAT: Debug purpose only.
// This function dumps a protobuf string format of HloModule.
static void DumpHloGraphForDebug(const std::vector<XlaCompiler::Argument>& args,
                                 std::unique_ptr<Graph> graph) {
  std::unique_ptr<FunctionLibraryDefinition> flib_def;
  std::unique_ptr<FunctionLibraryRuntime> flr;
  std::unique_ptr<XlaCompiler> compiler;

  xla::Client* client = xla::ClientLibrary::LocalClientOrDie();
  XlaOpRegistry::RegisterCompilationKernels();

  FunctionDefLibrary flib;
  flib_def.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));

  // Compiles the graph.
  XlaCompiler::Options options;
  DeviceType device_type("XLA_CPU_JIT");
  options.device_type = &device_type;
  options.client = client;
  options.flib_def = flib_def.get();
  compiler.reset(new XlaCompiler(options));

  // Compile graph
  XlaCompiler::CompilationResult result;
  TF_CHECK_OK(compiler->CompileGraph(XlaCompiler::CompileOptions(), "dump",
                                     std::move(graph), args, &result));

  // Convert to hlo
  xla::Computation& computation = *result.computation;

  xla::Service* service(
      static_cast<xla::Service*>(xla::ClientLibrary::GetXlaService(
          static_cast<xla::LocalClient*>(client)->platform())));
  const xla::ComputationTracker& computation_tracker =
      service->computation_tracker();

  auto user_computation_status =
      computation_tracker.Resolve(computation.handle());
  TF_CHECK_OK(user_computation_status.status());
  auto user_computation = user_computation_status.ConsumeValueOrDie();
  xla::VersionedComputationHandle versioned_handle =
      user_computation->GetVersionedHandle();
  std::unique_ptr<xla::HloModule> hlo_module =
      std::move(computation_tracker
                    .BuildHloModule(versioned_handle, xla::HloModuleConfig())
                    .ValueOrDie());
  VLOG(1) << "--- DUMP HLO ---";
  VLOG(1) << hlo_module->ToString();
}

TEST(XlaTfGraphUtil, ConvertTfGraphToSessionModule) {
  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args = BuildAddGraphArguments();
  std::unique_ptr<Graph> graph = BuildAddGraph();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::SessionModule> session_module,
      ConvertTfGraphToXlaSessionModule(args, std::move(graph)));

  ASSERT_EQ(5, session_module->entry().requests_size());

  VLOG(1) << "--- DUMP ---";
  VLOG(1) << session_module->DebugString();
  DumpHloGraphForDebug(args, BuildAddGraph());
}

TEST(XlaTfGraphUtil, ConvertXlaSessionModuleToXlaNodes) {
  std::vector<XlaCompiler::Argument> args = BuildAddGraphArguments();
  std::unique_ptr<Graph> graph = BuildAddGraph();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::SessionModule> session_module,
      ConvertTfGraphToXlaSessionModule(args, std::move(graph)));
  TF_ASSERT_OK_AND_ASSIGN(auto xla_nodes,
                          ConvertXlaSessionModuleToXlaNodes(*session_module));
  EXPECT_EQ(session_module->entry().requests_size(), xla_nodes.size());
}

}  // namespace xla_tf_graph
}  // namespace tensorflow
