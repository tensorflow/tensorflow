/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/jit/pjrt_compile_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace {

StatusOr<std::unique_ptr<Graph>> SampleGraphAddXY() {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  auto c = ops::Add(scope.WithOpName("C"), a, b);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));
  return graph;
}

StatusOr<FunctionDef> SampleFuntionAddXY(const std::string& name) {
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

TEST(PjrtCompileUtilTest, CompileToPjRtLoadedExecutable) {
  DeviceSetup device_setup;
  TF_ASSERT_OK_AND_ASSIGN(auto fdef, SampleFuntionAddXY("foo"));
  device_setup.AddDevicesAndSetUp({DEVICE_GPU}, fdef);

  Device* device = device_setup.GetDevice(DEVICE_GPU);
  const XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(device);

  NameAttrList function;
  function.set_name("foo");

  ResourceMgr resource_mgr("");

  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::PjRtLoadedExecutable* pjrt_executable = nullptr;
  xla::PjRtClient* pjrt_client = nullptr;

  TF_EXPECT_OK(CompileToPjRtLoadedExecutable(
      device, platform_info, function, SampleArgsForAddXY(),
      DeviceCompileMode::kStrict, /*has_ref_vars=*/true,
      /*may_alias_resource_update=*/true, device_setup.flr(), &resource_mgr,
      &compilation_result, &pjrt_client, &pjrt_executable));

  EXPECT_TRUE(compilation_result != nullptr);
  EXPECT_TRUE(pjrt_executable != nullptr);
  EXPECT_TRUE(pjrt_client != nullptr);
}

TEST(PjrtCompileUtilTest, CompileToPjRtLoadedExecutableWithOpKernelContext) {
  DeviceSetup device_setup;
  TF_ASSERT_OK_AND_ASSIGN(auto fdef, SampleFuntionAddXY("foo"));
  device_setup.AddDevicesAndSetUp({DEVICE_GPU}, fdef);

  Device* device = device_setup.GetDevice(DEVICE_GPU);
  const XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(device);

  NameAttrList function;
  function.set_name("foo");

  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  params.device = device;
  params.function_library = device_setup.flr();
  OpKernelContext ctx(&params, 1);

  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::PjRtLoadedExecutable* pjrt_executable = nullptr;
  xla::PjRtClient* pjrt_client = nullptr;

  TF_EXPECT_OK(CompileToPjRtLoadedExecutable(
      ctx, platform_info, function, SampleArgsForAddXY(),
      DeviceCompileMode::kStrict, /*has_ref_vars=*/true,
      /*may_alias_resource_update=*/true, &compilation_result, &pjrt_client,
      &pjrt_executable));

  EXPECT_TRUE(compilation_result != nullptr);
  EXPECT_TRUE(pjrt_executable != nullptr);
  EXPECT_TRUE(pjrt_client != nullptr);
}

}  // namespace
}  // namespace tensorflow
#endif
