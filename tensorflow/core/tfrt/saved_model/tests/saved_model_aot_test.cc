/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_aot_compile.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

// GPU target config for P100.
constexpr absl::string_view kGpuTargetConfig =
    R"pb(
  gpu_device_info {
    threads_per_block_limit: 1024
    threads_per_warp: 32
    shared_memory_per_block: 49152
    shared_memory_per_core: 65536
    threads_per_core_limit: 2048
    core_count: 56
    fpus_per_core: 64
    block_dim_limit_x: 2147483647
    block_dim_limit_y: 65535
    block_dim_limit_z: 65535
    memory_bandwidth: 732160000000
    l2_cache_size: 4194304
    clock_rate_ghz: 1.4805
    device_memory_size: 17066622976
    shared_memory_per_block_optin: 49152
    cuda_compute_capability { major: 6 }
  }
  platform_name: "CUDA"
  dnn_version_info {}
  device_description_str: "sm_6.0 with 17071734784B RAM, 56 cores, 1480500KHz clock, 715000KHz mem clock, 4194304B L2$"
    )pb";

absl::StatusOr<std::unique_ptr<Graph>> SampleGraphAddXY() {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  auto c = ops::Add(scope.WithOpName("C"), a, b);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));
  return graph;
}

absl::StatusOr<FunctionDef> SampleFunctionAddXY(const std::string& name) {
  TF_ASSIGN_OR_RETURN(auto graph, SampleGraphAddXY());
  FunctionDef fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph, name, &fdef));
  return fdef;
}

std::vector<XlaCompiler::Argument> SampleArgsForAddXY() {
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});
  return args;
}

StatusOr<NameAttrList> GetFunctionAndSetupDevice(DeviceSetup& device_setup) {
  TF_ASSIGN_OR_RETURN(auto fdef, SampleFunctionAddXY("foo"));
  device_setup.AddDevicesAndSetUp({DEVICE_GPU}, fdef);

  NameAttrList func_attr = NameAttrList();
  func_attr.set_name("foo");
  *func_attr.mutable_attr() = fdef.attr();
  return func_attr;
}

TEST(SavedModelAotTest, AotCompileTfFunctionToExecutable) {
  DeviceSetup device_setup;
  TF_ASSERT_OK_AND_ASSIGN(NameAttrList func_attr,
                          GetFunctionAndSetupDevice(device_setup));
  const FunctionLibraryDefinition* lib_def =
      device_setup.flr()->GetFunctionLibraryDefinition();

  XlaCompiler::CompilationResult compilation_result;
  XlaCompiler::CompilationResult* compilation_result_ptr = &compilation_result;
  stream_executor::GpuTargetConfigProto gpu_target_config_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(kGpuTargetConfig,
                                                  &gpu_target_config_proto));
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_executable,
      AotCompileToGpuPjRtExecutable(lib_def, func_attr, 0, SampleArgsForAddXY(),
                                    /*has_ref_vars=*/true,
                                    /*may_alias_resource_update=*/true,
                                    gpu_target_config_proto,
                                    &compilation_result_ptr));
  EXPECT_TRUE(compilation_result.computation != nullptr);
  EXPECT_TRUE(pjrt_executable != nullptr);
}

TEST(SavedModelAotTest, AotCompileTfFunctionToLoadedExecutableWithDevice) {
  DeviceSetup device_setup;
  TF_ASSERT_OK_AND_ASSIGN(NameAttrList func_attr,
                          GetFunctionAndSetupDevice(device_setup));
  const FunctionLibraryDefinition* lib_def =
      device_setup.flr()->GetFunctionLibraryDefinition();

  XlaCompiler::CompilationResult compilation_result;
  XlaCompiler::CompilationResult* compilation_result_ptr = &compilation_result;
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string serialized_pjrt_executable,
      AotCompileToGpuPjRtLoadedExecutableWithDevice(
          lib_def, func_attr, 0, SampleArgsForAddXY(),
          /*has_ref_vars=*/true,
          /*may_alias_resource_update=*/true, &compilation_result_ptr));
  EXPECT_TRUE(compilation_result.computation != nullptr);
  EXPECT_FALSE(serialized_pjrt_executable.empty());
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow

#endif
