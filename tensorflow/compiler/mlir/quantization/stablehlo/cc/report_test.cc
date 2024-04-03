/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/report.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace mlir::quant::stablehlo {
namespace {

using ::stablehlo::quantization::Method;
using ::stablehlo::quantization::QuantizableUnit;
using ::stablehlo::quantization::QuantizationResult;
using ::stablehlo::quantization::QuantizationResults;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::StrEq;

TEST(QuantizationReportTest, GetQuantizationResultsReturnsEmptyResults) {
  QuantizationReport report{};

  const QuantizationResults& results = report.GetQuantizationResults();
  ASSERT_THAT(results.results(), IsEmpty());
}

TEST(QuantizationReportTest, AddQuantizationResult) {
  // Construct a `QuantizationResult` to add, representing a unit named
  // `quantized_my_function` that is not quantized.
  QuantizationResult result{};
  QuantizableUnit& quantizable_unit = *result.mutable_quantizable_unit();
  quantizable_unit.set_name("quantized_my_function");

  Method& method = *result.mutable_method();
  method.mutable_no_quantization();

  QuantizationReport report{};
  report.AddQuantizationResult(std::move(result));

  const QuantizationResults& results = report.GetQuantizationResults();
  ASSERT_THAT(results.results(), SizeIs(1));

  const QuantizationResult& first_result = results.results(0);
  EXPECT_THAT(first_result.quantizable_unit().name(),
              StrEq("quantized_my_function"));
  EXPECT_TRUE(first_result.method().has_no_quantization());
}

}  // namespace
}  // namespace mlir::quant::stablehlo
