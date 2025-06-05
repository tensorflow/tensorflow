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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/testing/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"  // IWYU pragma: keep
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo::testing {

// NOLINTNEXTLINE - Automatically generated.
#define GEN_PASS_DEF_TESTLIFTQUANTIZABLESPOTSASFUNCTIONSWITHQUANTIZATIONSPECSPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/testing/passes.h.inc"

namespace {

using ::stablehlo::quantization::QuantizationSpecs;
using ::tsl::protobuf::TextFormat;
// NOLINTNEXTLINE(misc-include-cleaner) - Required for OSS.
using ::tsl::protobuf::io::ArrayInputStream;

// Empty (default) `QuantizationSpecs` proto.
constexpr absl::string_view kSpecsEmpty = R"pb(specs
                                               [])pb";

// Configure `QuantizationSpecs` to disable quantization for all dot_general
// quantizable units.
constexpr absl::string_view kSpecsDisableAllDotGeneral =
    R"pb(specs
         [ {
           matcher { function_name { regex: "composite_dot_general_.*" } }
           method { no_quantization {} }
         }])pb";

// Configure `QuantizationSpecs` to apply `StaticRangePtq` to all quantizable
// units.
constexpr absl::string_view kSpecsStaticRangePtqToAll =
    R"pb(specs
         [ {
           matcher { function_name { regex: ".*" } }
           method { static_range_ptq {} }
         }])pb";

// Configure `QuantizationSpecs` to apply `StaticRangePtq` to compute heavy
// units.
constexpr absl::string_view kSpecsStaticRangePtqToComputeHeavy =
    R"pb(specs
         [ {
           matcher { function_name { regex: "^.*(conv|dot|gather).*" } }
           method { static_range_ptq {} }
         }])pb";

class TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPass
    : public impl::
          TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPassBase<
              TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPass> {
 public:
  using impl::TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPassBase<
      TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPass>::
      TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPassBase;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPass)

 private:
  void runOnOperation() override;
};

// `TestQuantizationSpecs` -> predefined `QuantizationSpecs` textproto.
absl::string_view GetQuantizationSpecsTextProto(
    const TestQuantizationSpecs test_specs) {
  switch (test_specs) {
    case TestQuantizationSpecs::kEmpty:
      return kSpecsEmpty;
    case TestQuantizationSpecs::kDisableAllDotGeneral:
      return kSpecsDisableAllDotGeneral;
    case TestQuantizationSpecs::kStaticRangePtqToAll:
      return kSpecsStaticRangePtqToAll;
    case TestQuantizationSpecs::kStaticRangePtqToComputeHeavy:
      return kSpecsStaticRangePtqToComputeHeavy;
  }
}

// Parses a text proto into a `QuantizationSpecs` proto. Returns
// `InvalidArgumentError` if `text_proto` is invalid.
absl::StatusOr<QuantizationSpecs> ParseTextProto(
    const absl::string_view text_proto) {
  QuantizationSpecs quantization_specs;
  TextFormat::Parser parser;
  ArrayInputStream input_stream(text_proto.data(), text_proto.size());
  if (parser.Parse(&input_stream, &quantization_specs)) {
    return quantization_specs;
  }
  return absl::InvalidArgumentError("Could not parse text proto.");
}

void TestLiftQuantizableSpotsAsFunctionsWithQuantizationSpecsPass::
    runOnOperation() {
  PassManager pass_manager{&getContext()};

  // Construct `QuantizationSpecs` from the pass option `quantization-specs`.
  const absl::StatusOr<QuantizationSpecs> quantization_specs =
      ParseTextProto(GetQuantizationSpecsTextProto(quantization_specs_));
  if (!quantization_specs.ok()) {
    signalPassFailure();
    return;
  }

  pass_manager.addPass(
      quant::stablehlo::CreateLiftQuantizableSpotsAsFunctionsPass(
          *quantization_specs));

  if (failed(pass_manager.run(getOperation()))) {
    signalPassFailure();
  }
}

}  // namespace
}  // namespace mlir::quant::stablehlo::testing
