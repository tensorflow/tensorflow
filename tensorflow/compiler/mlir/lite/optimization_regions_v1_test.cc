/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// End-to-end tests of `odml.optimization_region` with optimize = "none" on
// the v1 converter (tf_tfl_passes.cc), driven exactly as jax_to_tfl_flatbuffer
// / tf_tfl_translate drive it: a StableHLO module through
// ConvertTFExecutorToTFLOrFlatbuffer with enable_hlo_to_tf_conversion.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

using ::testing::ElementsAre;

// How the quantization specs are set up; mirrors saved_model_to_tfl_flatbuffer
// for the `qdq_conversion_mode` converter flag.
enum class Mode { kDefault, kQDQStrict, kQDQDynamic, kQDQStatic };

const char* Name(Mode mode) {
  switch (mode) {
    case Mode::kDefault:
      return "default";
    case Mode::kQDQStrict:
      return "qdq strict";
    case Mode::kQDQDynamic:
      return "qdq dynamic";
    case Mode::kQDQStatic:
      return "qdq static";
  }
}

mlir::TFL::QuantizationSpecs QuantSpecs(Mode mode) {
  mlir::TFL::QuantizationSpecs quant_specs;
  switch (mode) {
    case Mode::kDefault:
      break;
    case Mode::kQDQStrict:
      quant_specs.qdq_conversion_mode =
          mlir::TFL::QDQConversionMode::kQDQStrict;
      break;
    case Mode::kQDQDynamic:
      quant_specs.qdq_conversion_mode =
          mlir::TFL::QDQConversionMode::kQDQDynamic;
      quant_specs.inference_type = tensorflow::DT_QINT8;
      break;
    case Mode::kQDQStatic:
      quant_specs.qdq_conversion_mode =
          mlir::TFL::QDQConversionMode::kQDQStatic;
      break;
  }
  if (mode != Mode::kDefault) {
    quant_specs.disable_set_input_nodes_quantization_params = true;
  }
  return quant_specs;
}

// How a body is wrapped: as the whole of @main (`kPlain`, the control) or as
// the decomposition of one `odml.optimization_region` with
// optimize = "none" (`kNone`).
struct Variant {
  std::string name;
  // The `composite_attributes` of the region, or empty for no region.
  std::string attrs;
};
const Variant kPlain{"plain", ""};
const Variant kNone{"none", R"(composite_attributes = {optimize = "none"}, )"};

// A non-splat f32 constant of `shape` as a `dense<...>` literal (a splat would
// be unfolded into a broadcast by the pipeline and is not what a weight is).
std::string DenseLiteral(llvm::ArrayRef<int64_t> shape) {
  int counter = 0;
  std::function<std::string(size_t)> dims = [&](size_t dim) {
    std::string text = "[";
    for (int64_t i = 0; i < shape[dim]; ++i) {
      if (i > 0) text += ", ";
      if (dim + 1 == shape.size()) {
        text += llvm::formatv("{0:f3}", 0.01f * (counter++ % 97)).str();
      } else {
        text += dims(dim + 1);
      }
    }
    return text + "]";
  };
  std::string type;
  for (int64_t d : shape) type += std::to_string(d) + "x";
  return "stablehlo.constant dense<" + dims(0) + "> : tensor<" + type + "f32>";
}

// `body` (StableHLO text over %arg0.. of `arg_types`, ending in
// `func.return`) as the whole of @main, or as the decomposition of one
// `odml.optimization_region` with `variant.attrs`.
std::string ProbeModule(const Variant& variant,
                        llvm::ArrayRef<std::string> arg_types,
                        llvm::StringRef result_type, llvm::StringRef body) {
  std::string args, operands, types;
  for (auto [i, type] : llvm::enumerate(arg_types)) {
    if (i > 0) args += ", ", operands += ", ", types += ", ";
    args += llvm::formatv("%arg{0}: {1}", i, type).str();
    operands += llvm::formatv("%arg{0}", i).str();
    types += type;
  }
  if (variant.attrs.empty()) {
    return llvm::formatv(
               "module {{\n  func.func @main({0}) -> {1} {{\n{2}\n  }\n}\n",
               args, result_type, body)
        .str();
  }
  return llvm::formatv(
             "module {{\n"
             "  func.func @main({0}) -> {1} {{\n"
             "    %0 = stablehlo.composite \"odml.optimization_region\" {2} {{ "
             "{3}decomposition = @body } : ({4}) -> {1}\n"
             "    func.return %0 : {1}\n"
             "  }\n"
             "  func.func private @body({0}) -> {1} {{\n{5}\n  }\n}\n",
             args, result_type, operands, variant.attrs, types, body)
      .str();
}

class V1RegionsTest : public ::testing::Test {
 protected:
  // Converts the StableHLO module `text` to a flatbuffer through the v1
  // converter, as jax_to_tfl_flatbuffer.cc / tf_tfl_translate do.
  absl::Status Convert(llvm::StringRef text, Mode mode, std::string* result) {
    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);
    mlir::RegisterAllTensorFlowDialects(registry);
    registry.insert<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
                    mlir::TFL::TensorFlowLiteDialect, mlir::mhlo::MhloDialect,
                    mlir::chlo::ChloDialect, mlir::quant::QuantDialect>();
    auto context = std::make_unique<mlir::MLIRContext>(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    if (!module)
      return absl::InvalidArgumentError("parse error:\n" + text.str());

    mlir::TFL::PassConfig pass_config(QuantSpecs(mode));
    pass_config.emit_builtin_tflite_ops = true;
    pass_config.enable_hlo_to_tf_conversion = true;
    pass_config.runtime_verification = true;
    pass_config.outline_tf_while = true;
    // The Python converter's default (lite.py `unfold_batchmatmul = False`),
    // which is what enables the batch_matmul -> fully_connected round.
    pass_config.unfold_batch_matmul = false;
    tflite::ConverterFlags converter_flags;
    return ConvertTFExecutorToTFLOrFlatbuffer(
        std::move(context), std::move(module), converter_flags, pass_config,
        /*saved_model_tags=*/{}, /*saved_model_dir=*/"", result,
        /*export_to_mlir=*/false);
  }

  // Converts the probe and returns the operators of its (single) subgraph as
  // `OP [in shapes] -> [out shapes]`, the shape operand of reshape / transpose
  // / expand_dims included, like the ai_edge_jax probes list them. Tensors
  // that are not f32 carry their type: `[8,8]:INT8`.
  std::vector<std::string> Ops(const Variant& variant,
                               llvm::ArrayRef<std::string> arg_types,
                               llvm::StringRef result_type,
                               llvm::StringRef body,
                               Mode mode = Mode::kDefault) {
    std::string text = ProbeModule(variant, arg_types, result_type, body);
    std::string buffer;
    absl::Status status = Convert(text, mode, &buffer);
    if (!status.ok()) return {"<conversion failed> " + status.ToString()};
    const tflite::Model* model = tflite::GetModel(buffer.data());
    EXPECT_EQ(model->subgraphs()->size(), 1u) << variant.name;
    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    auto shape = [&](int tensor) {
      std::string text = "[";
      const tflite::Tensor* t = subgraph->tensors()->Get(tensor);
      const auto* dims = t->shape();
      for (unsigned i = 0; dims && i < dims->size(); ++i) {
        if (i > 0) text += ",";
        text += std::to_string(dims->Get(i));
      }
      text += "]";
      if (t->type() != tflite::TensorType_FLOAT32) {
        text += std::string(":") + tflite::EnumNameTensorType(t->type());
      }
      return text;
    };
    std::vector<std::string> ops;
    for (const tflite::Operator* op : *subgraph->operators()) {
      std::string text = tflite::EnumNameBuiltinOperator(tflite::GetBuiltinCode(
          model->operator_codes()->Get(op->opcode_index())));
      for (int input : *op->inputs()) {
        if (input >= 0) text += " " + shape(input);
      }
      text += " ->";
      for (int output : *op->outputs()) text += " " + shape(output);
      ops.push_back(text);
    }
    return ops;
  }
};

////////////////////////////////////////////////////////////////////////////////
// The direct-lowering probes
////////////////////////////////////////////////////////////////////////////////

// The "rank-transformation probes" of the ai_edge_jax region tests
// (cl/991083670, ops/test/test_region.py, test_probe_*): five author-written
// TFL patterns that the converter rewrites, each as the frontend emits it
// (`custom_call @tfl.*` carriers), on its own (the rewrite happens) and in an
// optimize = "none" region (the author's ops are kept, explicit zero bias
// included).

// P1: batch_matmul with a constant 2-D rhs becomes a fully_connected
// (OptimizeBatchMatmul). An activation rhs is left alone.
TEST_F(V1RegionsTest, BmmWithConstantRhsBecomesFullyConnected) {
  const std::string body =
      "    %w = " + DenseLiteral({16, 32}) + "\n" +
      R"(    %out = stablehlo.custom_call @tfl.batch_matmul(%arg0, %w) {backend_config = "{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}"} : (tensor<1x4x8x16xf32>, tensor<16x32xf32>) -> tensor<1x4x8x32xf32>
    func.return %out : tensor<1x4x8x32xf32>)";
  EXPECT_THAT(
      Ops(kPlain, {"tensor<1x4x8x16xf32>"}, "tensor<1x4x8x32xf32>", body),
      ElementsAre("FULLY_CONNECTED [1,4,8,16] [32,16] -> [1,4,8,32]"));
  EXPECT_THAT(
      Ops(kNone, {"tensor<1x4x8x16xf32>"}, "tensor<1x4x8x32xf32>", body),
      ElementsAre("BATCH_MATMUL [1,4,8,16] [16,32] -> [1,4,8,32]"));

  const std::string activation_rhs =
      R"(    %out = stablehlo.custom_call @tfl.batch_matmul(%arg0, %arg1) {backend_config = "{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}"} : (tensor<1x4x8x16xf32>, tensor<1x4x16x32xf32>) -> tensor<1x4x8x32xf32>
    func.return %out : tensor<1x4x8x32xf32>)";
  for (const Variant& variant : {kPlain, kNone}) {
    EXPECT_THAT(
        Ops(variant, {"tensor<1x4x8x16xf32>", "tensor<1x4x16x32xf32>"},
            "tensor<1x4x8x32xf32>", activation_rhs),
        ElementsAre("BATCH_MATMUL [1,4,8,16] [1,4,16,32] -> [1,4,8,32]"))
        << variant.name;
  }
}

// P2: a unary op after an author reshape is hoisted in front of it
// (ReorderElementwiseAndMoveOperations). A binary op with a constant operand
// is not moved.
TEST_F(V1RegionsTest, UnaryOpAfterReshapeIsHoisted) {
  const std::string exp =
      R"(    %s = stablehlo.constant dense<[1, 8, 4, 16]> : tensor<4xi32>
    %r = stablehlo.custom_call @tfl.reshape(%arg0, %s) {backend_config = "{}"} : (tensor<1x8x64xf32>, tensor<4xi32>) -> tensor<1x8x4x16xf32>
    %out = stablehlo.custom_call @tfl.exp(%r) {backend_config = "{}"} : (tensor<1x8x4x16xf32>) -> tensor<1x8x4x16xf32>
    func.return %out : tensor<1x8x4x16xf32>)";
  EXPECT_THAT(Ops(kPlain, {"tensor<1x8x64xf32>"}, "tensor<1x8x4x16xf32>", exp),
              ElementsAre("EXP [1,8,64] -> [1,8,64]",
                          "RESHAPE [1,8,64] [4]:INT32 -> [1,8,4,16]"));
  EXPECT_THAT(Ops(kNone, {"tensor<1x8x64xf32>"}, "tensor<1x8x4x16xf32>", exp),
              ElementsAre("RESHAPE [1,8,64] [4]:INT32 -> [1,8,4,16]",
                          "EXP [1,8,4,16] -> [1,8,4,16]"));

  const std::string mul =
      "    %c = " + DenseLiteral({1, 1, 4, 16}) + "\n" +
      R"(    %s = stablehlo.constant dense<[1, 8, 4, 16]> : tensor<4xi32>
    %r = stablehlo.custom_call @tfl.reshape(%arg0, %s) {backend_config = "{}"} : (tensor<1x8x64xf32>, tensor<4xi32>) -> tensor<1x8x4x16xf32>
    %out = stablehlo.custom_call @tfl.mul(%r, %c) {backend_config = "{fused_activation_function = \"NONE\"}"} : (tensor<1x8x4x16xf32>, tensor<1x1x4x16xf32>) -> tensor<1x8x4x16xf32>
    func.return %out : tensor<1x8x4x16xf32>)";
  for (const Variant& variant : {kPlain, kNone}) {
    EXPECT_THAT(
        Ops(variant, {"tensor<1x8x64xf32>"}, "tensor<1x8x4x16xf32>", mul),
        ElementsAre("RESHAPE [1,8,64] [4]:INT32 -> [1,8,4,16]",
                    "MUL [1,8,4,16] [1,1,4,16] -> [1,8,4,16]"))
        << variant.name;
  }
}

// P3: expand_dims and a transpose that only moves unit dimensions become
// reshapes (ConvertExpandDimsToReshape, ConvertTrivialTransposeOpToReshapeOp).
TEST_F(V1RegionsTest, ExpandDimsAndTrivialTransposeBecomeReshape) {
  const std::string expand =
      R"(    %d = stablehlo.constant dense<2> : tensor<i32>
    %out = stablehlo.custom_call @tfl.expand_dims(%arg0, %d) {backend_config = "{}"} : (tensor<1x8x16xf32>, tensor<i32>) -> tensor<1x8x1x16xf32>
    func.return %out : tensor<1x8x1x16xf32>)";
  EXPECT_THAT(
      Ops(kPlain, {"tensor<1x8x16xf32>"}, "tensor<1x8x1x16xf32>", expand),
      ElementsAre("RESHAPE [1,8,16] [4]:INT32 -> [1,8,1,16]"));
  EXPECT_THAT(
      Ops(kNone, {"tensor<1x8x16xf32>"}, "tensor<1x8x1x16xf32>", expand),
      ElementsAre("EXPAND_DIMS [1,8,16] []:INT32 -> [1,8,1,16]"));

  const std::string transpose =
      R"(    %p = stablehlo.constant dense<[1, 0, 2, 3]> : tensor<4xi32>
    %out = stablehlo.custom_call @tfl.transpose(%arg0, %p) {backend_config = "{}"} : (tensor<1x8x1x16xf32>, tensor<4xi32>) -> tensor<8x1x1x16xf32>
    func.return %out : tensor<8x1x1x16xf32>)";
  EXPECT_THAT(
      Ops(kPlain, {"tensor<1x8x1x16xf32>"}, "tensor<8x1x1x16xf32>", transpose),
      ElementsAre("RESHAPE [1,8,1,16] [4]:INT32 -> [8,1,1,16]"));
  EXPECT_THAT(
      Ops(kNone, {"tensor<1x8x1x16xf32>"}, "tensor<8x1x1x16xf32>", transpose),
      ElementsAre("TRANSPOSE [1,8,1,16] [4]:INT32 -> [8,1,1,16]"));
}

// P4: an explicit flatten to 2-D in front of a fully_connected. The probe
// lists the reshape as kept; here, with keep_num_dims = false, the converter
// removes it (RemoveReshapeBeforeFullyConnected) and lets the fully_connected
// flatten the input itself. The FC's zero bias (ai_edge_jax emits a zeros
// constant for a missing bias) is dropped by the TFL dialect's own
// canonicalizer (RemoveOptionalZeroBias) wherever the body is canonicalized;
// only the `none` body keeps it, as emitted. A full-rank splat constant
// operand of a mul keeps its rank under every variant.
TEST_F(V1RegionsTest, ReshapeBeforeFullyConnected) {
  const std::string body =
      "    %w = " + DenseLiteral({32, 16}) + "\n" +
      R"(    %b = stablehlo.constant dense<0.0> : tensor<32xf32>
    %s = stablehlo.constant dense<[8, 16]> : tensor<2xi32>
    %r = stablehlo.custom_call @tfl.reshape(%arg0, %s) {backend_config = "{}"} : (tensor<1x8x1x16xf32>, tensor<2xi32>) -> tensor<8x16xf32>
    %out = stablehlo.custom_call @tfl.fully_connected(%r, %w, %b) {backend_config = "{fused_activation_function = \"NONE\", keep_num_dims = false, asymmetric_quantize_inputs = false, weights_format = \"DEFAULT\"}"} : (tensor<8x16xf32>, tensor<32x16xf32>, tensor<32xf32>) -> tensor<8x32xf32>
    func.return %out : tensor<8x32xf32>)";
  EXPECT_THAT(Ops(kPlain, {"tensor<1x8x1x16xf32>"}, "tensor<8x32xf32>", body),
              ElementsAre("FULLY_CONNECTED [1,8,1,16] [32,16] -> [8,32]"));
  EXPECT_THAT(Ops(kNone, {"tensor<1x8x1x16xf32>"}, "tensor<8x32xf32>", body),
              ElementsAre("RESHAPE [1,8,1,16] [2]:INT32 -> [8,16]",
                          "FULLY_CONNECTED [8,16] [32,16] [32] -> [8,32]"));

  const std::string splat =
      R"(    %c = stablehlo.constant dense<0.5> : tensor<1x8x4x16xf32>
    %out = stablehlo.custom_call @tfl.mul(%arg0, %c) {backend_config = "{fused_activation_function = \"NONE\"}"} : (tensor<1x8x4x16xf32>, tensor<1x8x4x16xf32>) -> tensor<1x8x4x16xf32>
    func.return %out : tensor<1x8x4x16xf32>)";
  for (const Variant& variant : {kPlain, kNone}) {
    EXPECT_THAT(
        Ops(variant, {"tensor<1x8x4x16xf32>"}, "tensor<1x8x4x16xf32>", splat),
        ElementsAre("MUL [1,8,4,16] [1,8,4,16] -> [1,8,4,16]"))
        << variant.name;
  }
}

// P5: a separate 4-D bias add after a fully_connected is folded into the FC
// bias (UndoBroadcastFullyConnectedBiasAdd + FuseFullyConnectedAndAdd). With
// keep_num_dims = false the FC is also switched to keep_num_dims and a
// reshape produces the broadcast result.
TEST_F(V1RegionsTest, FullyConnectedThenBroadcastBiasAdd) {
  auto body = [&](llvm::StringRef keep_num_dims, llvm::StringRef fc_type,
                  llvm::StringRef out_type) {
    return "    %w = " + DenseLiteral({32, 16}) +
           "\n    %b4 = " + DenseLiteral({1, 1, 1, 32}) + "\n" +
           "    %b = stablehlo.constant dense<0.0> : tensor<32xf32>\n"
           "    %fc = stablehlo.custom_call @tfl.fully_connected(%arg0, %w, "
           "%b) "
           "{backend_config = \"{fused_activation_function = \\\"NONE\\\", "
           "keep_num_dims = " +
           keep_num_dims.str() +
           ", asymmetric_quantize_inputs = false, weights_format = "
           "\\\"DEFAULT\\\"}\"} : (tensor<1x8x1x16xf32>, tensor<32x16xf32>, "
           "tensor<32xf32>) -> " +
           fc_type.str() +
           "\n"
           "    %out = stablehlo.custom_call @tfl.add(%fc, %b4) "
           "{backend_config "
           "= \"{fused_activation_function = \\\"NONE\\\"}\"} : (" +
           fc_type.str() + ", tensor<1x1x1x32xf32>) -> " + out_type.str() +
           "\n"
           "    func.return %out : " +
           out_type.str();
  };
  // keep_num_dims = true: one 4-D FC.
  const std::string keep =
      body("true", "tensor<1x8x1x32xf32>", "tensor<1x8x1x32xf32>");
  EXPECT_THAT(
      Ops(kPlain, {"tensor<1x8x1x16xf32>"}, "tensor<1x8x1x32xf32>", keep),
      ElementsAre("FULLY_CONNECTED [1,8,1,16] [32,16] [32] -> [1,8,1,32]"));
  EXPECT_THAT(
      Ops(kNone, {"tensor<1x8x1x16xf32>"}, "tensor<1x8x1x32xf32>", keep),
      ElementsAre("FULLY_CONNECTED [1,8,1,16] [32,16] [32] -> [1,8,1,32]",
                  "ADD [1,8,1,32] [1,1,1,32] -> [1,8,1,32]"));
  // keep_num_dims = false: the add broadcasts [8,32] against [1,1,1,32].
  const std::string flat =
      body("false", "tensor<8x32xf32>", "tensor<1x1x8x32xf32>");
  EXPECT_THAT(
      Ops(kPlain, {"tensor<1x8x1x16xf32>"}, "tensor<1x1x8x32xf32>", flat),
      ElementsAre("FULLY_CONNECTED [1,8,1,16] [32,16] [32] -> [1,8,1,32]",
                  "RESHAPE [1,8,1,32] [4]:INT32 -> [1,1,8,32]"));
  EXPECT_THAT(
      Ops(kNone, {"tensor<1x8x1x16xf32>"}, "tensor<1x1x8x32xf32>", flat),
      ElementsAre("FULLY_CONNECTED [1,8,1,16] [32,16] [32] -> [8,32]",
                  "ADD [8,32] [1,1,1,32] -> [1,1,8,32]"));
}

////////////////////////////////////////////////////////////////////////////////
// Explicit quantize / dequantize
////////////////////////////////////////////////////////////////////////////////

// A fully annotated fully_connected the way a frontend emits explicit
// quantization: Q/DQ on the activation, dequantized int8 weights, Q/DQ on the
// output. Outside a region the quantizer fuses this (in every QDQ mode) into
// one int8 fully_connected between a quantize and a dequantize (in the
// static mode the boundary Q/DQ are folded into int8 model inputs and
// outputs as well); in the default mode no quantization pass runs and the
// ops stay as written, minus the zero bias that canonicalization drops. In an
// optimize = "none" region they stay as written in every mode: Q and DQ remain
// two ops with the authored types, the fully_connected runs in float with its
// zero bias, and the weights reach it through a dequantize.
constexpr char kAnnotatedFc[] =
    R"(    %q = stablehlo.custom_call @tfl.quantize(%arg0) {backend_config = "{scale = [5.000000e-01 : f32], zero_point = [3 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xf32>) -> tensor<1x4x8xi8>
    %dq = stablehlo.custom_call @tfl.dequantize(%q) {backend_config = "{scale = [5.000000e-01 : f32], zero_point = [3 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xi8>) -> tensor<1x4x8xf32>
    %w = stablehlo.constant dense<3> : tensor<8x8xi8>
    %dw = stablehlo.custom_call @tfl.dequantize(%w) {backend_config = "{scale = [3.125000e-02 : f32], zero_point = [0 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<8x8xi8>) -> tensor<8x8xf32>
    %b = stablehlo.constant dense<0.0> : tensor<8xf32>
    %fc = stablehlo.custom_call @tfl.fully_connected(%dq, %dw, %b) {backend_config = "{fused_activation_function = \"NONE\", keep_num_dims = true, asymmetric_quantize_inputs = false, weights_format = \"DEFAULT\"}"} : (tensor<1x4x8xf32>, tensor<8x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
    %q2 = stablehlo.custom_call @tfl.quantize(%fc) {backend_config = "{scale = [2.500000e-01 : f32], zero_point = [0 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xf32>) -> tensor<1x4x8xi8>
    %out = stablehlo.custom_call @tfl.dequantize(%q2) {backend_config = "{scale = [2.500000e-01 : f32], zero_point = [0 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xi8>) -> tensor<1x4x8xf32>
    func.return %out : tensor<1x4x8xf32>)";

const std::vector<std::string> kLiteral = {
    "QUANTIZE [1,4,8] -> [1,4,8]:INT8",
    "DEQUANTIZE [1,4,8]:INT8 -> [1,4,8]",
    "DEQUANTIZE [8,8]:INT8 -> [8,8]",
    "FULLY_CONNECTED [1,4,8] [8,8] [8] -> [1,4,8]",
    "QUANTIZE [1,4,8] -> [1,4,8]:INT8",
    "DEQUANTIZE [1,4,8]:INT8 -> [1,4,8]",
};
const std::vector<std::string> kLiteralWithoutZeroBias = {
    "QUANTIZE [1,4,8] -> [1,4,8]:INT8",
    "DEQUANTIZE [1,4,8]:INT8 -> [1,4,8]",
    "DEQUANTIZE [8,8]:INT8 -> [8,8]",
    "FULLY_CONNECTED [1,4,8] [8,8] -> [1,4,8]",
    "QUANTIZE [1,4,8] -> [1,4,8]:INT8",
    "DEQUANTIZE [1,4,8]:INT8 -> [1,4,8]",
};
const std::vector<std::string> kFused = {
    "QUANTIZE [1,4,8] -> [1,4,8]:INT8",
    "FULLY_CONNECTED [1,4,8]:INT8 [8,8]:INT8 -> [1,4,8]:INT8",
    "DEQUANTIZE [1,4,8]:INT8 -> [1,4,8]",
};
const std::vector<std::string> kFusedInt8Io = {
    "FULLY_CONNECTED [1,4,8]:INT8 [8,8]:INT8 -> [1,4,8]:INT8",
};

TEST_F(V1RegionsTest, ExplicitQdqIsLiteralInNoneRegions) {
  for (Mode mode : {Mode::kDefault, Mode::kQDQStrict, Mode::kQDQDynamic,
                    Mode::kQDQStatic}) {
    EXPECT_THAT(Ops(kNone, {"tensor<1x4x8xf32>"}, "tensor<1x4x8xf32>",
                    kAnnotatedFc, mode),
                ::testing::ElementsAreArray(kLiteral))
        << Name(mode);
  }
}

TEST_F(V1RegionsTest, ExplicitQdqOutsideRegionsIsUnchanged) {
  // The control: today's behaviour. The default mode runs no quantization
  // pass, so the annotations survive as ops (only the zero bias goes); every
  // QDQ mode fuses them.
  EXPECT_THAT(Ops(kPlain, {"tensor<1x4x8xf32>"}, "tensor<1x4x8xf32>",
                  kAnnotatedFc, Mode::kDefault),
              ::testing::ElementsAreArray(kLiteralWithoutZeroBias));
  for (Mode mode : {Mode::kQDQStrict, Mode::kQDQDynamic}) {
    EXPECT_THAT(Ops(kPlain, {"tensor<1x4x8xf32>"}, "tensor<1x4x8xf32>",
                    kAnnotatedFc, mode),
                ::testing::ElementsAreArray(kFused))
        << Name(mode);
  }
  EXPECT_THAT(Ops(kPlain, {"tensor<1x4x8xf32>"}, "tensor<1x4x8xf32>",
                  kAnnotatedFc, Mode::kQDQStatic),
              ::testing::ElementsAreArray(kFusedInt8Io));
}

////////////////////////////////////////////////////////////////////////////////
// Errors
////////////////////////////////////////////////////////////////////////////////

TEST_F(V1RegionsTest, UnsupportedRegionAttributeFailsConversion) {
  const Variant disable{
      "disable", R"(composite_attributes = {disable = ["bmm_to_fc"]}, )"};
  const std::string body =
      R"(    %out = stablehlo.custom_call @tfl.exp(%arg0) {backend_config = "{}"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
    func.return %out : tensor<4x8xf32>)";
  std::string buffer;
  absl::Status status = Convert(
      ProbeModule(disable, {"tensor<4x8xf32>"}, "tensor<4x8xf32>", body),
      Mode::kDefault, &buffer);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("only optimize = \"none\" is supported"));
}

}  // namespace
}  // namespace tensorflow
