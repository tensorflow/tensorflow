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

#include "tensorflow/compiler/mlir/tf2xla/internal/test_matchers.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/service/hlo.pb.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tsl/platform/statusor.h"

namespace {
using ::tensorflow::monitoring::testing::CellReader;
using ::testing::Not;

constexpr char kMetric[] = "/tensorflow/metric";
auto* counter =
    tensorflow::monitoring::Counter<1>::New(kMetric, "description", "status");
constexpr char kOkStatus[] = "ok";
const int kArbitraryIntResult = 37;

template <typename T>
tsl::StatusOr<T> success(T t) {
  return t;
}
absl::StatusOr<int> success() { return kArbitraryIntResult; }
template <typename T>
tsl::StatusOr<T> filtered(T t) {
  return tsl::StatusOr<T>(tensorflow::CompileToHloGraphAnalysisFailedError());
}
absl::StatusOr<int> filtered() { return filtered(kArbitraryIntResult); }
absl::StatusOr<int> failed() {
  return absl::StatusOr<int>(absl::InternalError("fail"));
}

TEST(TestUtil, MatchesOk) { ASSERT_THAT(success(), IsOkOrFiltered()); }

TEST(TestUtil, DoesntMatchesFailure) {
  ASSERT_THAT(failed(), Not(IsOkOrFiltered()));
}

TEST(TestUtil, MatchesFiltered) { ASSERT_THAT(filtered(), IsOkOrFiltered()); }

TEST(TestUtil, IncrementsOk) {
  CellReader<int64_t> reader(kMetric);
  counter->GetCell(kOkStatus)->IncrementBy(1);

  ASSERT_THAT(success(), IncrementedOrFiltered(reader.Delta(kOkStatus), 1));
}

TEST(TestUtil, FilteredDoesntIncrementsOk) {
  CellReader<int64_t> reader(kMetric);

  ASSERT_THAT(filtered(), IncrementedOrFiltered(reader.Delta(kOkStatus), 1));
}

TEST(TestUtil, FailureDoesntMatchIncrement) {
  CellReader<int64_t> reader(kMetric);

  ASSERT_THAT(failed(), Not(IncrementedOrFiltered(reader.Delta(kOkStatus), 1)));
}

tensorflow::XlaCompilationResult CreateXlaComputationResult(
    const char* hlo_name) {
  auto result = tensorflow::XlaCompilationResult();
  xla::HloModuleProto hlo;
  hlo.set_name(hlo_name);
  result.computation = std::make_shared<xla::XlaComputation>(hlo);
  return result;
}

TEST(TestUtil, ComputationContainsOk) {
  constexpr char arbitrary_hlo[] = "arbitrary_hlo";
  auto result = CreateXlaComputationResult(arbitrary_hlo);

  ASSERT_THAT(success(result), ComputationProtoContains(arbitrary_hlo));
}

TEST(TestUtil, ComputationDoesNotContain) {
  constexpr char arbitrary_hlo[] = "arbitrary_hlo";
  constexpr char bad_hlo[] = "bad_hlo";
  auto result = CreateXlaComputationResult(arbitrary_hlo);

  ASSERT_THAT(success(result), Not(ComputationProtoContains(bad_hlo)));
}

TEST(TestUtil, ComputationDoesNotContainFiltered) {
  constexpr char arbitrary_hlo[] = "arbitrary_hlo";
  constexpr char bad_hlo[] = "bad_hlo";
  auto result = CreateXlaComputationResult(arbitrary_hlo);

  ASSERT_THAT(filtered(result), ComputationProtoContains(bad_hlo));
}

TEST(TestUtil, MlirModuleHas) {
  constexpr char arbirary_mlir[] = "arbirary_mlir";

  ASSERT_THAT(success(arbirary_mlir), HasMlirModuleWith(arbirary_mlir));
}

TEST(TestUtil, MlirModuleDoesNotHave) {
  constexpr char arbirary_mlir[] = "arbirary_mlir";
  constexpr char bad_mlir[] = "bad_mlir";

  ASSERT_THAT(success(arbirary_mlir), Not(HasMlirModuleWith(bad_mlir)));
}

TEST(TestUtil, MlirModuleDoesNotHaveFiltered) {
  constexpr char arbirary_mlir[] = "arbirary_mlir";
  constexpr char bad_mlir[] = "bad_mlir";

  ASSERT_THAT(filtered(arbirary_mlir), HasMlirModuleWith(bad_mlir));
}

}  // namespace
