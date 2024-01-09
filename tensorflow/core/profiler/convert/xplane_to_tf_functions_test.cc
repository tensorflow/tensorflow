/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/tf_function.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_test_utils.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/utils/tf_xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kEager = "eager";
const absl::string_view kConcrete = "concrete";
const absl::string_view kTracedNonXla = "traced-nonXla";
const absl::string_view kTracedXla = "traced-xla";
const absl::string_view kNotTracedNonXla = "notTraced-nonXla";
const absl::string_view kNotTracedXla = "notTraced-xla";

constexpr double kMaxError = 0.001;

TfFunctionDb ConvertXSpaceToTfFunctionDb(const XSpace& space) {
  TfFunctionDb result;
  const XPlane* host_plane = FindPlaneWithName(space, kHostThreadsPlaneName);
  if (host_plane) {
    XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_plane);
    plane.ForEachLine([&result](const XLineVisitor& line) {
      TfFunctionDb tf_function_db = ConvertHostThreadsXLineToTfFunctionDb(line);
      CombineTfFunctionDb(tf_function_db, &result);
    });
  }
  return result;
}

TEST(ConvertXPlaneToTfFunctions, CombineTwoThreads) {
  XSpace space;
  XPlaneBuilder host_plane_builder(space.add_planes());
  host_plane_builder.SetName(kHostThreadsPlaneName);
  host_plane_builder.ReserveLines(2);
  std::string kFunctionName = "decrement";

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread, kFunctionName,
                            10, 100, kTracedNonXla, 1);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread, kFunctionName,
                            150, 20, kNotTracedNonXla, 2);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread, kFunctionName,
                            200, 80, kTracedNonXla, 3);

  auto other_thread = host_plane_builder.GetOrCreateLine(1);
  CreateTfFunctionCallEvent(&host_plane_builder, &other_thread, kFunctionName,
                            20, 100, kTracedNonXla, 2);
  CreateTfFunctionCallEvent(&host_plane_builder, &other_thread, kFunctionName,
                            160, 20, kNotTracedNonXla, 2);
  CreateTfFunctionCallEvent(&host_plane_builder, &other_thread, kFunctionName,
                            210, 80, kTracedXla, 4);

  TfFunctionDb tf_function_db = ConvertXSpaceToTfFunctionDb(space);
  EXPECT_EQ(tf_function_db.tf_functions().size(), 1);
  EXPECT_EQ(tf_function_db.tf_functions().count(kFunctionName), 1);
  const TfFunction& tf_function =
      tf_function_db.tf_functions().at(kFunctionName);
  EXPECT_EQ(tf_function.total_tracing_count(), 4);
  EXPECT_EQ(tf_function.compiler(), MIXED_COMPILER);
  EXPECT_NEAR(tf_function.expensive_call_percent(), 90, kMaxError);

  const auto& metrics = tf_function.metrics();
  EXPECT_EQ(metrics.size(), 2);
  EXPECT_EQ(metrics.count(TRACED_MODE), 1);
  EXPECT_EQ(metrics.count(NOT_TRACED_MODE), 1);
  const auto& traced_mode = metrics.at(TRACED_MODE);
  EXPECT_EQ(traced_mode.count(), 4);
  EXPECT_EQ(traced_mode.self_time_ps(), 360);
  const auto& not_traced_mode = metrics.at(NOT_TRACED_MODE);
  EXPECT_EQ(not_traced_mode.count(), 2);
  EXPECT_EQ(not_traced_mode.self_time_ps(), 40);
}

TEST(ConvertXPlaneToTfFunctions, NestedFunctions) {
  XSpace space;
  XPlaneBuilder host_plane_builder(space.add_planes());
  host_plane_builder.SetName(kHostThreadsPlaneName);
  host_plane_builder.ReserveLines(1);
  std::string kOuterFunctionName = "outer";
  std::string kInnerFunctionName = "inner";

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread,
                            kOuterFunctionName, 10, 100, kTracedNonXla, 1);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread,
                            kInnerFunctionName, 30, 40, kNotTracedXla, 0);
  TfFunctionDb tf_function_db = ConvertXSpaceToTfFunctionDb(space);
  EXPECT_EQ(tf_function_db.tf_functions().size(), 2);
  EXPECT_EQ(tf_function_db.tf_functions().count(kOuterFunctionName), 1);
  EXPECT_EQ(tf_function_db.tf_functions().count(kInnerFunctionName), 1);
  const TfFunction& outer =
      tf_function_db.tf_functions().at(kOuterFunctionName);
  EXPECT_EQ(outer.total_tracing_count(), 1);
  EXPECT_EQ(outer.compiler(), OTHER_COMPILER);
  EXPECT_NEAR(outer.expensive_call_percent(), 100, kMaxError);
  const auto& outer_metrics = outer.metrics();
  EXPECT_EQ(outer_metrics.size(), 1);
  EXPECT_EQ(outer_metrics.count(TRACED_MODE), 1);
  const auto& traced_mode = outer_metrics.at(TRACED_MODE);
  EXPECT_EQ(traced_mode.count(), 1);
  EXPECT_EQ(traced_mode.self_time_ps(), 60);
  const TfFunction& inner =
      tf_function_db.tf_functions().at(kInnerFunctionName);
  EXPECT_EQ(inner.total_tracing_count(), 0);
  EXPECT_EQ(inner.compiler(), XLA_COMPILER);
  EXPECT_NEAR(inner.expensive_call_percent(), 0, kMaxError);
  const auto& inner_metrics = inner.metrics();
  EXPECT_EQ(inner_metrics.size(), 1);
  EXPECT_EQ(inner_metrics.count(NOT_TRACED_MODE), 1);
  const auto& not_traced_mode = inner_metrics.at(NOT_TRACED_MODE);
  EXPECT_EQ(not_traced_mode.count(), 1);
  EXPECT_EQ(not_traced_mode.self_time_ps(), 40);
}

TEST(ConvertXPlaneToTfFunctions, EagerPlusConcrete) {
  XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  host_plane_builder.ReserveLines(2);
  std::string kEagerFunctionName = "i_am_eager";
  std::string kConcreteFunctionName = "i_am_concrete";

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateTfFunctionCallEvent(&host_plane_builder, &main_thread,
                            kEagerFunctionName, 10, 200, kEager);
  auto other_thread = host_plane_builder.GetOrCreateLine(1);
  CreateTfFunctionCallEvent(&host_plane_builder, &other_thread,
                            kConcreteFunctionName, 20, 40, kConcrete);
  TfFunctionDb tf_function_db = ConvertXSpaceToTfFunctionDb(space);
  EXPECT_EQ(tf_function_db.tf_functions().size(), 2);
  EXPECT_EQ(tf_function_db.tf_functions().count(kEagerFunctionName), 1);
  EXPECT_EQ(tf_function_db.tf_functions().count(kConcreteFunctionName), 1);
  const TfFunction& eager =
      tf_function_db.tf_functions().at(kEagerFunctionName);
  EXPECT_EQ(eager.total_tracing_count(), 0);
  EXPECT_EQ(eager.compiler(), INVALID_COMPILER);
  EXPECT_NEAR(eager.expensive_call_percent(), 100, kMaxError);
  const auto& eager_metrics = eager.metrics();
  EXPECT_EQ(eager_metrics.size(), 1);
  EXPECT_EQ(eager_metrics.count(EAGER_MODE), 1);
  const auto& eager_mode = eager_metrics.at(EAGER_MODE);
  EXPECT_EQ(eager_mode.count(), 1);
  EXPECT_EQ(eager_mode.self_time_ps(), 200);
  const TfFunction& concrete =
      tf_function_db.tf_functions().at(kConcreteFunctionName);
  EXPECT_EQ(concrete.total_tracing_count(), 0);
  EXPECT_EQ(concrete.compiler(), INVALID_COMPILER);
  EXPECT_NEAR(concrete.expensive_call_percent(), 0, kMaxError);
  const auto& concrete_metrics = concrete.metrics();
  EXPECT_EQ(concrete_metrics.size(), 1);
  EXPECT_EQ(concrete_metrics.count(CONCRETE_MODE), 1);
  const auto& concrete_mode = concrete_metrics.at(CONCRETE_MODE);
  EXPECT_EQ(concrete_mode.count(), 1);
  EXPECT_EQ(concrete_mode.self_time_ps(), 40);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
