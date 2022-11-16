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

#include "tensorflow/compiler/jit/device_compilation_profiler.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/jit/tests/xla_compilation_cache_test_helper.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"

namespace tensorflow {
namespace {

TEST(DeviceCompilationProfilerTest, RegisterExecution) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  for (int i = 0; i < 5; ++i) {
    profiler->RegisterExecution(function);
  }
  TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
  EXPECT_EQ(stats.execution_count, 5);
}

TEST(DeviceCompilationProfilerTest, RegisterCompilation) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  auto listener = std::make_unique<JitCompilationListener>();
  auto listener_ptr = listener.get();
  RegisterXlaActivityListener(std::move(listener));

  NameAttrList function;
  function.set_name("TestFunc");

  std::vector<XlaJitCompilationActivity> expected_activities;
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(profiler->RegisterCompilation(function, 4, false).ok());

    TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
    XlaJitCompilationActivity expected_activity;
    expected_activity.set_cluster_name(function.name());
    expected_activity.set_compile_count(stats.compile_count);
    expected_activity.set_compile_time_us(4);
    expected_activity.set_cumulative_compile_time_us(
        stats.cumulative_compile_time_us);
    expected_activity.set_used_persistent_cache(false);
    expected_activities.push_back(expected_activity);
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
  EXPECT_EQ(stats.compile_count, 5);
  EXPECT_EQ(stats.cumulative_compile_time_us, 5 * 4);

  // TODO(b/255826209): Use ::testing::EqualsProto once b/135192747 is fixed.
  const auto& actual_activities = listener_ptr->GetListenerHistory();
  EXPECT_EQ(actual_activities.size(), expected_activities.size());
  for (size_t i = 0; i < actual_activities.size(); ++i) {
    EXPECT_EQ(actual_activities[i].SerializeAsString(),
              expected_activities[i].SerializeAsString());
  }
}

TEST(DeviceCompilationProfilerTest, OngoingAsyncCompilations) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  for (int i = 0; i < 5; ++i) {
    profiler->IncrementOngoingAsyncCompilations();
  }

  EXPECT_EQ(profiler->GetNumOngoingAsyncCompilations(), 5);

  for (int i = 0; i < 5; ++i) {
    profiler->DecrementOngoingAsyncCompilations();
  }

  EXPECT_EQ(profiler->GetNumOngoingAsyncCompilations(), 0);

  for (int i = 0; i < 5; ++i) {
    profiler->IncrementOngoingAsyncCompilations();
    profiler->DecrementOngoingAsyncCompilations();
  }

  EXPECT_EQ(profiler->GetNumOngoingAsyncCompilations(), 0);
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterNotFound) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kStrict, 0));
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterFirstExecution) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  profiler->RegisterExecution(function);

  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterMegamorphic) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  const int64_t kCompileThreshold = 10;
  const int64_t kMinExecutionsPerCompile = 50;

  // Register compilation enough times (without registering executions enough
  // times) so that the function is marked megamorphic.
  for (int i = 0; i < kCompileThreshold + 1; ++i) {
    EXPECT_TRUE(profiler->RegisterCompilation(function, 1, false).ok());
  }
  profiler->RegisterExecution(function);

  // Shouldn't compile cluster since it has gone megamorphic.
  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));
  TF_ASSERT_OK_AND_ASSIGN(auto stats, profiler->GetCompileStats(function));
  EXPECT_TRUE(stats.is_megamorphic);

  // Always compile for strict compile mode.
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kStrict, 0));

  // Once a cluster has gone megamorphic, it remains megamorphic (even though
  // it's being executed more frequently now) and shouldn't be compiled again.
  for (int i = 0; i < kCompileThreshold * kMinExecutionsPerCompile + 1; ++i) {
    profiler->RegisterExecution(function);
  }

  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));
  TF_ASSERT_OK_AND_ASSIGN(stats, profiler->GetCompileStats(function));
  EXPECT_TRUE(stats.is_megamorphic);

  // Always compile for strict compile mode.
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kStrict, 0));
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterAsync) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  const int64_t kMaxNumOngoingCompilations = 10;
  for (int i = 0; i < kMaxNumOngoingCompilations; ++i) {
    profiler->IncrementOngoingAsyncCompilations();
  }

  // Should allow compilation since this is the first execution.
  profiler->RegisterExecution(function);
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));

  // Should not allow compilation since this is not the first execution and
  // we've already reached the maximum number of ongoing compilations allowed.
  profiler->RegisterExecution(function);
  EXPECT_FALSE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));

  profiler->DecrementOngoingAsyncCompilations();
  // Should allow compilation since we've decremented the number of ongoing
  // compilations.
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kAsync, 0));
}

TEST(DeviceCompilationProfilerTest, ShouldCompileClusterLazy) {
  DeviceCompilationProfiler* profiler = new DeviceCompilationProfiler();
  core::ScopedUnref profiler_ref(profiler);

  NameAttrList function;
  function.set_name("TestFunc");

  constexpr int64_t kDefaultCompilationThreshold = 2;

  // Should allow compilation since this is the first execution.
  profiler->RegisterExecution(function);
  EXPECT_TRUE(
      profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy, 0));

  // Shouldn't allow compilation until compilation has been requested at least
  // kDefaultCompilationThreshold times.
  profiler->RegisterExecution(function);
  for (int current_request_count = 0;
       current_request_count < kDefaultCompilationThreshold;
       ++current_request_count) {
    EXPECT_FALSE(profiler->ShouldCompileCluster(
        function, DeviceCompileMode::kLazy, current_request_count));
  }
  EXPECT_TRUE(profiler->ShouldCompileCluster(function, DeviceCompileMode::kLazy,
                                             kDefaultCompilationThreshold));
}

}  // namespace
}  // namespace tensorflow
