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

#include "tensorflow/compiler/jit/device_compiler.h"

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include "tensorflow/compiler/jit/device_compiler_client.h"
#include "tensorflow/compiler/jit/tests/device_compiler_test_helper.h"
#include "tensorflow/compiler/jit/xla_device_compiler_client.h"
#include "xla/client/client_library.h"
#include "xla/stream_executor/platform_manager.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace {
using ::testing::_;
using ::testing::Return;

using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
using XlaDeviceExecutablePersistor =
    DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>;
using Signature = DeviceCompilationClusterSignature;

xla::LocalClient* GetLocalClient() {
  // TODO(b/255826209): Figure out how to run this test with the CPU client as
  // well.
  auto platform = se::PlatformManager::PlatformWithName("cuda").value();
  return xla::ClientLibrary::GetOrCreateLocalClient(platform).value();
}

XlaDeviceCompiler* CreateXlaDeviceCompiler(bool enable_persistence = false) {
  auto xla_compiler_client =
      std::make_unique<XlaDeviceCompilerClient>(GetLocalClient());
  auto xla_persistor = std::make_unique<XlaDeviceExecutablePersistor>(
      XlaDeviceExecutablePersistor::Config{
          enable_persistence ? testing::TmpDir() : "", false, "xla"},
      DeviceType(DEVICE_GPU_XLA_JIT));
  return new XlaDeviceCompiler(std::move(xla_persistor),
                               std::move(xla_compiler_client));
}

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

absl::StatusOr<FunctionDef> SampleFuntionAddXY(const std::string& name) {
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

class MockXlaDeviceExecutablePersistor
    : public DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient> {
 public:
  MockXlaDeviceExecutablePersistor()
      : DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>(
            Config{testing::TmpDir(), false, "xla"},
            DeviceType(DEVICE_CPU_XLA_JIT)) {}
  MOCK_METHOD(absl::Status, TryToPersistExecutable,
              (uint64, const std::string&, const XlaCompiler::Options&,
               const XlaCompiler::CompilationResult&,
               const xla::LocalExecutable&,
               (DeviceCompilerClient<xla::LocalExecutable, xla::LocalClient>*)),
              (const, override));
};

class MockDeviceCompilationProfiler : public DeviceCompilationProfiler {
 public:
  MOCK_METHOD(bool, ShouldCompileCluster,
              (const NameAttrList& function, DeviceCompileMode compile_mode,
               int64_t current_request_count),
              (override));
  MOCK_METHOD(absl::Status, RegisterCompilation,
              (const NameAttrList& function, int64_t compile_time_us,
               bool used_persistent_cache),
              (override));
};

class DeviceCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    flib_def_ = std::make_unique<FunctionLibraryDefinition>(
        OpRegistry::Global(), FunctionDefLibrary());
    TF_ASSERT_OK_AND_ASSIGN(auto fdef, SampleFuntionAddXY("foo"));
    TF_ASSERT_OK(flib_def_->AddFunctionDef(fdef));

    profiler_ = new DeviceCompilationProfiler();
    profiler_ref_ = std::make_unique<core::ScopedUnref>(profiler_);

    mock_profiler_ = new MockDeviceCompilationProfiler();
    mock_profiler_ref_ = std::make_unique<core::ScopedUnref>(mock_profiler_);

    xla_device_compiler_ = CreateXlaDeviceCompiler();
    xla_device_compiler_ref_ =
        std::make_unique<core::ScopedUnref>(xla_device_compiler_);

    auto listener = std::make_unique<JitCompilationListener>();
    listener_ = listener.get();
    RegisterXlaActivityListener(std::move(listener));
  }

  XlaCompiler::Options GetDefaultXlaOptions() {
    XlaCompiler::Options options;
    options.device_type = DeviceType(DEVICE_GPU_XLA_JIT);
    options.client = xla_device_compiler_->client();
    options.flib_def = flib_def_.get();
    return options;
  }

  absl::StatusOr<std::unique_ptr<xla::LocalExecutable>>
  BuildSampleXlaExecutable() {
    TF_ASSIGN_OR_RETURN(auto graph, SampleGraphAddXY());
    auto args = SampleArgsForAddXY();

    // Compiles the graph.
    XlaCompiler compiler(GetDefaultXlaOptions());

    XlaCompiler::CompilationResult compilation_result;
    TF_RETURN_IF_ERROR(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                             "graph", std::move(graph), args,
                                             &compilation_result));
    return xla_device_compiler_->compiler_client()->BuildExecutable(
        GetDefaultXlaOptions(), compilation_result);
  }

  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  JitCompilationListener* listener_;

  DeviceCompilationProfiler* profiler_;
  std::unique_ptr<core::ScopedUnref> profiler_ref_;

  MockDeviceCompilationProfiler* mock_profiler_;
  std::unique_ptr<core::ScopedUnref> mock_profiler_ref_;

  XlaDeviceCompiler* xla_device_compiler_;
  std::unique_ptr<core::ScopedUnref> xla_device_compiler_ref_;
};

TEST_F(DeviceCompilerTest, CompileStrictSuccess) {
  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::LocalExecutable* xla_executable = nullptr;

  XlaCompiler::Options options = GetDefaultXlaOptions();

  NameAttrList fn;
  fn.set_name("foo");

  TF_EXPECT_OK(xla_device_compiler_->CompileIfNeeded(
      options, fn, SampleArgsForAddXY(), XlaCompiler::CompileOptions{},
      DeviceCompileMode::kStrict, profiler_, &compilation_result,
      &xla_executable));

  EXPECT_TRUE(compilation_result != nullptr);
  EXPECT_TRUE(xla_executable != nullptr);
}

TEST_F(DeviceCompilerTest, CompileShouldCompileClusterFalse) {
  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::LocalExecutable* xla_executable = nullptr;

  XlaCompiler::Options options = GetDefaultXlaOptions();

  NameAttrList fn;
  fn.set_name("foo");

  // Using a mock here since it's difficult to have ShouldCompileCluster()
  // return false.
  EXPECT_CALL(*mock_profiler_,
              ShouldCompileCluster(_, DeviceCompileMode::kLazy, 1))
      .WillOnce(Return(false));

  TF_EXPECT_OK(xla_device_compiler_->CompileIfNeeded(
      options, fn, SampleArgsForAddXY(), XlaCompiler::CompileOptions{},
      DeviceCompileMode::kLazy, mock_profiler_, &compilation_result,
      &xla_executable));

  EXPECT_TRUE(compilation_result == nullptr);
  EXPECT_TRUE(xla_executable == nullptr);
}

TEST_F(DeviceCompilerTest, CompileCacheHit) {
  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::LocalExecutable* xla_executable = nullptr;

  XlaCompiler::Options options = GetDefaultXlaOptions();

  NameAttrList fn;
  fn.set_name("foo");

  TF_EXPECT_OK(xla_device_compiler_->CompileIfNeeded(
      options, fn, SampleArgsForAddXY(), XlaCompiler::CompileOptions{},
      DeviceCompileMode::kStrict, profiler_, &compilation_result,
      &xla_executable));

  EXPECT_TRUE(compilation_result != nullptr);
  EXPECT_TRUE(xla_executable != nullptr);

  const XlaCompiler::CompilationResult* new_compilation_result = nullptr;
  xla::LocalExecutable* new_xla_executable = nullptr;

  // Request compiling the same function again.
  TF_EXPECT_OK(xla_device_compiler_->CompileIfNeeded(
      options, fn, SampleArgsForAddXY(), XlaCompiler::CompileOptions{},
      DeviceCompileMode::kStrict, profiler_, &new_compilation_result,
      &new_xla_executable));

  // new_compilation_result and new_xla_executable should point to the
  // compilation_result and executable returned after the first compilation
  // request.
  EXPECT_EQ(compilation_result, new_compilation_result);
  EXPECT_EQ(xla_executable, new_xla_executable);
}

TEST_F(DeviceCompilerTest, CompileAsyncSuccess) {
  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::LocalExecutable* xla_executable = nullptr;

  XlaCompiler::Options options = GetDefaultXlaOptions();

  NameAttrList fn;
  fn.set_name("foo");

  // Using a mock here to determine when the async compilation finishes. This is
  // to avoid using absl::SleepFor().
  // `RegisterCompilation` is the last call that happens just before the async
  // compilation completes. We use the completion of this call to determine when
  // the compilation finshes to verify expected behavior.
  Notification done;
  EXPECT_CALL(*mock_profiler_,
              ShouldCompileCluster(_, DeviceCompileMode::kAsync, 1))
      .WillOnce(Return(true));
  EXPECT_CALL(*mock_profiler_, RegisterCompilation(_, _, false))
      .WillOnce([&done] {
        done.Notify();
        return absl::OkStatus();
      });

  auto args = SampleArgsForAddXY();
  TF_EXPECT_OK(xla_device_compiler_->CompileIfNeeded(
      options, fn, args, XlaCompiler::CompileOptions{},
      DeviceCompileMode::kAsync, mock_profiler_, &compilation_result,
      &xla_executable));

  // compilation_result and xla_executable aren't available immediately after
  // requesting compilation in asynchronous mode.
  EXPECT_TRUE(compilation_result == nullptr);
  EXPECT_TRUE(xla_executable == nullptr);

  // Check if an appropriate entry is made in xla_cache.
  auto xla_cache = xla_device_compiler_->cache();
  TF_ASSERT_OK_AND_ASSIGN(auto signature, Signature::Build(fn, args));

  auto cache_value = xla_cache->Lookup(signature);
  EXPECT_TRUE(cache_value);
  EXPECT_TRUE(cache_value->compile_state != DeviceCompileState::kUncompiled);

  // Wait for async compilation to complete.
  done.WaitForNotification();
  cache_value = xla_cache->Lookup(signature);
  EXPECT_TRUE(cache_value);
  EXPECT_TRUE(cache_value->compile_state == DeviceCompileState::kCompiled);
  EXPECT_TRUE(cache_value->compilation_result != nullptr);
  EXPECT_TRUE(cache_value->executable != nullptr);
  EXPECT_TRUE(cache_value->compilation_status.ok());
}

TEST_F(DeviceCompilerTest, CompilePersistentCacheEnabled) {
  auto xla_device_compiler =
      CreateXlaDeviceCompiler(/*enable_persistence=*/true);
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  NameAttrList fn;
  fn.set_name("foo");
  auto args = SampleArgsForAddXY();
  XlaCompiler::Options options = GetDefaultXlaOptions();

  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::LocalExecutable* xla_executable = nullptr;

  TF_EXPECT_OK(xla_device_compiler->CompileIfNeeded(
      options, fn, args, XlaCompiler::CompileOptions{},
      DeviceCompileMode::kStrict, profiler_, &compilation_result,
      &xla_executable));

  EXPECT_TRUE(compilation_result != nullptr);
  EXPECT_TRUE(xla_executable != nullptr);

  // Check if device_compiler was able to load the executable from the
  // persistent cache.
  std::vector<XlaJitCompilationActivity> activity_history =
      listener_->GetListenerHistory();
  EXPECT_EQ(activity_history.size(), 1);
  EXPECT_EQ(activity_history[0].cluster_name(), fn.name());
  EXPECT_EQ(activity_history[0].compile_count(), 1);
  EXPECT_FALSE(activity_history[0].used_persistent_cache());

  listener_->ClearListenerHistory();

  // Create another DeviceCompiler object pointing to the same persistent cache
  // directory. It should load the executable instead of building it.
  auto xla_device_compiler_2 =
      CreateXlaDeviceCompiler(/*enable_persistence=*/true);
  core::ScopedUnref xla_device_compiler_ref_2(xla_device_compiler_2);

  auto profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  const XlaCompiler::CompilationResult* compilation_result_2 = nullptr;
  xla::LocalExecutable* xla_executable_2 = nullptr;
  TF_EXPECT_OK(xla_device_compiler_2->CompileIfNeeded(
      options, fn, args, XlaCompiler::CompileOptions{},
      DeviceCompileMode::kStrict, profiler, &compilation_result_2,
      &xla_executable_2));

  EXPECT_TRUE(compilation_result_2 != nullptr);
  EXPECT_TRUE(xla_executable_2 != nullptr);

  activity_history = listener_->GetListenerHistory();
  EXPECT_EQ(activity_history.size(), 1);
  EXPECT_EQ(activity_history[0].cluster_name(), fn.name());
  EXPECT_EQ(activity_history[0].compile_count(), 1);
  // Verify that the executable was loaded instead of built.
  EXPECT_TRUE(activity_history[0].used_persistent_cache());
}

TEST_F(DeviceCompilerTest, CompileFailedToLoadFromPersistentCache) {
  auto xla_device_compiler =
      CreateXlaDeviceCompiler(/*enable_persistence=*/true);
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  NameAttrList fn;
  fn.set_name("foo");
  auto args = SampleArgsForAddXY();
  XlaCompiler::Options options = GetDefaultXlaOptions();

  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::LocalExecutable* xla_executable = nullptr;

  // Persist an executable.
  TF_EXPECT_OK(xla_device_compiler->CompileIfNeeded(
      options, fn, args, XlaCompiler::CompileOptions{},
      DeviceCompileMode::kStrict, profiler_, &compilation_result,
      &xla_executable));

  // Corrupt the file which contains the serialized executable.
  std::vector<string> files;
  TF_ASSERT_OK(Env::Default()->GetChildren(testing::TmpDir(), &files));
  std::string const* serialized_executable_filename = nullptr;
  for (const auto& file : files) {
    if (absl::StartsWith(file, "xla__")) {
      serialized_executable_filename = &file;
      break;
    }
  }
  EXPECT_TRUE(serialized_executable_filename != nullptr);
  std::string serialized_executable_filepath =
      io::JoinPath(testing::TmpDir(), *serialized_executable_filename);
  std::unique_ptr<WritableFile> serialized_executable_file;
  TF_ASSERT_OK(Env::Default()->NewWritableFile(serialized_executable_filepath,
                                               &serialized_executable_file));
  TF_ASSERT_OK(serialized_executable_file->Append("Garbage."));
  TF_ASSERT_OK(serialized_executable_file->Close());

  // Create another DeviceCompiler object pointing to the same persistent cache
  // directory. It should error out while loading the executable from the
  // corrupt file.
  auto xla_device_compiler_2 =
      CreateXlaDeviceCompiler(/*enable_persistence=*/true);
  core::ScopedUnref xla_device_compiler_ref_2(xla_device_compiler_2);

  const XlaCompiler::CompilationResult* compilation_result_2 = nullptr;
  xla::LocalExecutable* xla_executable_2 = nullptr;

  EXPECT_FALSE(xla_device_compiler_2
                   ->CompileIfNeeded(options, fn, args,
                                     XlaCompiler::CompileOptions{},
                                     DeviceCompileMode::kStrict, profiler_,
                                     &compilation_result_2, &xla_executable_2)
                   .ok());

  EXPECT_TRUE(compilation_result_2 == nullptr);
  EXPECT_TRUE(xla_executable_2 == nullptr);
}

TEST_F(DeviceCompilerTest, CompileStrictPersistentCacheFailedToPersist) {
  auto xla_compiler_client =
      std::make_unique<XlaDeviceCompilerClient>(GetLocalClient());
  auto xla_persistor = std::make_unique<MockXlaDeviceExecutablePersistor>();
  auto xla_device_compiler = new XlaDeviceCompiler(
      std::move(xla_persistor), std::move(xla_compiler_client));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  NameAttrList fn;
  fn.set_name("foo");
  auto args = SampleArgsForAddXY();
  XlaCompiler::Options options = GetDefaultXlaOptions();

  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::LocalExecutable* xla_executable = nullptr;

  auto persistor = down_cast<MockXlaDeviceExecutablePersistor*>(
      xla_device_compiler->persistor());
  TF_ASSERT_OK_AND_ASSIGN(auto signature, Signature::Build(fn, args));
  EXPECT_CALL(*persistor,
              TryToPersistExecutable(Signature::Hash()(signature),
                                     signature.HumanString(), _, _, _, _))
      .WillOnce(Return(errors::FailedPrecondition("Random error.")));

  EXPECT_THAT(xla_device_compiler->CompileIfNeeded(
                  options, fn, args, XlaCompiler::CompileOptions{},
                  DeviceCompileMode::kStrict, profiler_, &compilation_result,
                  &xla_executable),
              testing::StatusIs(error::FAILED_PRECONDITION,
                                ::testing::HasSubstr("Random error.")));

  EXPECT_TRUE(compilation_result == nullptr);
  EXPECT_TRUE(xla_executable == nullptr);
}

TEST_F(OpsTestBase, CompileSingleOpSuccess) {
  TF_EXPECT_OK(NodeDefBuilder("identity_op", "Identity")
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DT_FLOAT)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2}), {6.9, 4.2});
  TF_EXPECT_OK(RunOpKernel());

  auto xla_device_compiler = CreateXlaDeviceCompiler();
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  auto profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  const XlaCompiler::CompilationResult* compilation_result = nullptr;
  xla::LocalExecutable* xla_executable = nullptr;

  XlaOpRegistry::RegisterCompilationKernels();
  auto flib_def = std::make_unique<FunctionLibraryDefinition>(
      OpRegistry::Global(), FunctionDefLibrary());

  XlaCompiler::Options options;
  options.device_type = DeviceType(DEVICE_GPU_XLA_JIT);
  options.client = GetLocalClient();
  options.flib_def = flib_def.get();

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_FLOAT;
  args[0].shape = TensorShape({1, 2});
  args[0].constant_value = GetInput(0);
  args[0].initialized = true;

  NameAttrList fn;
  fn.set_name("foo");

  TF_EXPECT_OK(xla_device_compiler->CompileSingleOpIfNeeded(
      options, args, XlaCompiler::CompileOptions{}, context_.get(), profiler,
      &compilation_result, &xla_executable));

  EXPECT_TRUE(compilation_result != nullptr);
  EXPECT_TRUE(xla_executable != nullptr);
}

}  // namespace
}  // namespace tensorflow
