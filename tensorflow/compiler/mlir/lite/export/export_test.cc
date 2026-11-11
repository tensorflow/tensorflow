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

#include "tensorflow/compiler/mlir/lite/export/export.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"

namespace mlir::TFL {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

class ExportTest : public ::testing::Test {
 protected:
  ExportTest() {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, quant::QuantDialect,
                    stablehlo::StablehloDialect, vhlo::VhloDialect,
                    TensorFlowLiteDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  OwningOpRef<ModuleOp> Parse(absl::string_view mlir) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        llvm::StringRef(mlir.data(), mlir.size()), &context_);
    EXPECT_TRUE(module);
    return module;
  }

  absl::Status Export(ModuleOp module, std::string& file) {
    llvm::SmallString<1024> buffer;
    llvm::raw_svector_ostream stream(buffer);
    absl::Status status = ExportFlatbuffer(module, stream);
    file = buffer.str().str();
    return status;
  }

  // Exports `mlir` and returns the model file.
  std::string ExportOrDie(absl::string_view mlir) {
    OwningOpRef<ModuleOp> module = Parse(mlir);
    std::string file;
    absl::Status status = Export(*module, file);
    EXPECT_TRUE(status.ok()) << status;
    return file;
  }

  MLIRContext context_;
};

// Returns the bytes of the buffer of `tensor`.
absl::string_view GetData(absl::string_view file, const tflite::Model* model,
                          const tflite::Tensor* tensor) {
  const tflite::Buffer* buffer = model->buffers()->Get(tensor->buffer());
  return file.substr(buffer->offset(), buffer->size());
}

template <typename T>
std::vector<T> GetValues(absl::string_view data) {
  std::vector<T> values(data.size() / sizeof(T));
  std::memcpy(values.data(), data.data(), data.size());
  return values;
}

std::string GetMetadata(absl::string_view file, const tflite::Model* model,
                        absl::string_view name) {
  for (const tflite::Metadata* metadata : *model->metadata()) {
    if (metadata->name()->string_view() != name) continue;
    const tflite::Buffer* buffer = model->buffers()->Get(metadata->buffer());
    return std::string(reinterpret_cast<const char*>(buffer->data()->data()),
                       buffer->data()->size());
  }
  return "";
}

TEST_F(ExportTest, StoresBuffersAfterFlatbuffer) {
  const std::string file = ExportOrDie(R"mlir(
    func.func @main() -> tensor<2xi32> {
      %cst = "tfl.pseudo_const"() {value = dense<[7, 8]> : tensor<2xi32>} : () -> tensor<2xi32>
      func.return %cst : tensor<2xi32>
    }
  )mlir");
  const tflite::Model* model = tflite::GetModel(file.data());

  EXPECT_EQ(GetMetadata(file, model, "buffer_location"), "outside flatbuffers");
  const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
  EXPECT_EQ(subgraph->name()->str(), "main");
  const tflite::Buffer* buffer =
      model->buffers()->Get(subgraph->tensors()->Get(0)->buffer());
  EXPECT_EQ(buffer->offset() % 16, 0);
  EXPECT_GT(buffer->offset(), 0);
  EXPECT_THAT(
      GetValues<int32_t>(GetData(file, model, subgraph->tensors()->Get(0))),
      ElementsAre(7, 8));
}

TEST_F(ExportTest, SharesConstantsAcrossSignaturesAndReleasesBlobs) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    module attributes {tf_saved_model.semantics} {
      func.func @prefill(%x: tensor<4xf32> {tf_saved_model.index_path = ["x"]})
          -> (tensor<4xf32> {tf_saved_model.index_path = ["y"]})
          attributes {tf.entry_function = {inputs = "prefill_x", outputs = "prefill_y"},
                      tf_saved_model.exported_names = ["prefill"]} {
        %w = "tfl.pseudo_const"() {value = dense_resource<w0> : tensor<4xf32>} : () -> tensor<4xf32>
        %0 = "tfl.mul"(%x, %w) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        func.return %0 : tensor<4xf32>
      }
      func.func @decode(%x: tensor<4xf32> {tf_saved_model.index_path = ["x"]})
          -> (tensor<4xf32> {tf_saved_model.index_path = ["y"]})
          attributes {tf.entry_function = {inputs = "decode_x", outputs = "decode_y"},
                      tf_saved_model.exported_names = ["decode"]} {
        %w = "tfl.pseudo_const"() {value = dense_resource<w1> : tensor<4xf32>} : () -> tensor<4xf32>
        %0 = "tfl.mul"(%x, %w) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
        func.return %0 : tensor<4xf32>
      }
    }
    {-#
      dialect_resources: {
        builtin: {
          w0: "0x040000000000803F000000400000404000008040",
          w1: "0x040000000000803F000000400000404000008040"
        }
      }
    #-}
  )mlir");
  std::string file;
  ASSERT_TRUE(Export(*module, file).ok());
  const tflite::Model* model = tflite::GetModel(file.data());

  ASSERT_EQ(model->subgraphs()->size(), 2);
  const tflite::Tensor* w0 = model->subgraphs()->Get(0)->tensors()->Get(1);
  const tflite::Tensor* w1 = model->subgraphs()->Get(1)->tensors()->Get(1);
  EXPECT_EQ(w0->buffer(), w1->buffer());
  EXPECT_THAT(GetValues<float>(GetData(file, model, w0)),
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));

  ASSERT_EQ(model->signature_defs()->size(), 2);
  const tflite::SignatureDef* decode = model->signature_defs()->Get(1);
  EXPECT_EQ(decode->signature_key()->str(), "decode");
  EXPECT_EQ(decode->subgraph_index(), 1);
  EXPECT_EQ(decode->inputs()->Get(0)->name()->str(), "x");
  EXPECT_EQ(model->subgraphs()->Get(1)->tensors()->Get(0)->name()->str(),
            "decode_x");

  // The weights are released once written.
  module->walk([](Operation* op) {
    if (auto constant = dyn_cast<ConstOp>(op)) {
      auto resource = cast<DenseResourceElementsAttr>(constant.getValue());
      EXPECT_TRUE(resource.getRawHandle().getBlob()->getData().empty());
    }
  });
}

TEST_F(ExportTest, FoldsCastOfBf16ConstantIntoF32Buffer) {
  const std::string file = ExportOrDie(R"mlir(
    func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
      %cst = "tfl.pseudo_const"() {value = dense<[1.5, -2.0]> : tensor<2xbf16>} : () -> tensor<2xbf16> loc("weight")
      %cast = "tfl.cast"(%cst) : (tensor<2xbf16>) -> tensor<2xf32>
      %0 = "tfl.add"(%arg0, %cast) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
      func.return %0 : tensor<2xf32>
    }
  )mlir");
  const tflite::Model* model = tflite::GetModel(file.data());

  const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
  ASSERT_EQ(subgraph->tensors()->size(), 3);
  ASSERT_EQ(subgraph->operators()->size(), 1);
  const tflite::Tensor* weight = subgraph->tensors()->Get(1);
  EXPECT_EQ(weight->name()->str(), "weight");
  EXPECT_EQ(weight->type(), tflite::TensorType_FLOAT32);
  EXPECT_THAT(GetValues<float>(GetData(file, model, weight)),
              ElementsAre(1.5f, -2.0f));
}

TEST_F(ExportTest, PacksInt4Constants) {
  const std::string file = ExportOrDie(R"mlir(
    func.func @main() -> tensor<4xi4> {
      %cst = "tfl.pseudo_const"() {value = dense<[1, 2, 3, -1]> : tensor<4xi4>} : () -> tensor<4xi4>
      func.return %cst : tensor<4xi4>
    }
  )mlir");
  const tflite::Model* model = tflite::GetModel(file.data());

  const tflite::Tensor* tensor = model->subgraphs()->Get(0)->tensors()->Get(0);
  EXPECT_EQ(tensor->type(), tflite::TensorType_INT4);
  EXPECT_THAT(GetValues<uint8_t>(GetData(file, model, tensor)),
              ElementsAre(0x21, 0xF3));
}

TEST_F(ExportTest, StoresCustomOptionsAfterFlatbuffer) {
  const std::string file = ExportOrDie(R"mlir(
    func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
      %0 = "tfl.custom"(%arg0) {custom_code = "my_op", custom_option = #tfl<const_bytes : "0x0102030405">} : (tensor<2xf32>) -> tensor<2xf32>
      func.return %0 : tensor<2xf32>
    }
  )mlir");
  const tflite::Model* model = tflite::GetModel(file.data());

  const tflite::Operator* op = model->subgraphs()->Get(0)->operators()->Get(0);
  const tflite::OperatorCode* opcode =
      model->operator_codes()->Get(op->opcode_index());
  EXPECT_EQ(opcode->builtin_code(), tflite::BuiltinOperator_CUSTOM);
  EXPECT_EQ(opcode->custom_code()->str(), "my_op");
  EXPECT_EQ(op->large_custom_options_offset() % 16, 0);
  EXPECT_EQ(absl::string_view(file).substr(op->large_custom_options_offset(),
                                           op->large_custom_options_size()),
            "\x01\x02\x03\x04\x05");
}

TEST_F(ExportTest, ExportsDynamicShapesAsShapeSignatures) {
  const std::string file = ExportOrDie(R"mlir(
    func.func @main(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
      %0 = "tfl.relu"(%arg0) : (tensor<?x4xf32>) -> tensor<?x4xf32>
      func.return %0 : tensor<?x4xf32>
    }
  )mlir");
  const tflite::Model* model = tflite::GetModel(file.data());

  EXPECT_THAT(GetMetadata(file, model, "min_runtime_version"), HasSubstr("."));
  const tflite::Tensor* tensor = model->subgraphs()->Get(0)->tensors()->Get(0);
  EXPECT_THAT(
      std::vector<int32_t>(tensor->shape()->begin(), tensor->shape()->end()),
      ElementsAre(1, 4));
  EXPECT_THAT(std::vector<int32_t>(tensor->shape_signature()->begin(),
                                   tensor->shape_signature()->end()),
              ElementsAre(-1, 4));
}

TEST_F(ExportTest, ExportsBlockwiseQuantizationParametersAsTensors) {
  const std::string file = ExportOrDie(R"mlir(
    func.func @main() -> tensor<4x4xf32> {
      %w = "tfl.pseudo_qconst"() {qtype = tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>, value = dense<1> : tensor<4x4xi8>} : () -> tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>
      %0 = "tfl.dequantize"(%w) : (tensor<4x4x!tfl.blockwise_quant<i8 -> f32, block_shape = [2, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>) -> tensor<4x4xf32>
      func.return %0 : tensor<4x4xf32>
    }
  )mlir");
  const tflite::Model* model = tflite::GetModel(file.data());

  const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
  const tflite::Tensor* weight = subgraph->tensors()->Get(0);
  EXPECT_EQ(weight->type(), tflite::TensorType_INT8);
  const tflite::BlockwiseQuantization* blockwise =
      weight->quantization()->details_as_BlockwiseQuantization();
  ASSERT_NE(blockwise, nullptr);
  EXPECT_EQ(blockwise->zero_points(), -1);
  const tflite::Tensor* scales = subgraph->tensors()->Get(blockwise->scales());
  EXPECT_EQ(blockwise->scales(), 1);
  EXPECT_EQ(scales->name()->str(), weight->name()->str() + "_scales");
  EXPECT_THAT(GetValues<float>(GetData(file, model, scales)),
              ElementsAre(0.5f, 0.25f));
}

TEST_F(ExportTest, ExportsCompositeWithDecompositionSubgraph) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    func.func private @decomposition(%x: tensor<2xf32>) -> tensor<2xf32> {
      return %x : tensor<2xf32>
    }
    func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
      %0 = stablehlo.composite "odml.identity" %x {
          composite_attributes = {n = 3 : i64, name = "id"},
          decomposition = @decomposition} : (tensor<2xf32>) -> tensor<2xf32>
      return %0 : tensor<2xf32>
    }
  )mlir");
  PassManager pm(&context_);
  pm.addPass(odml::createLegalizeStablehloToVhloPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  std::string file;
  ASSERT_TRUE(Export(*module, file).ok());
  const tflite::Model* model = tflite::GetModel(file.data());

  ASSERT_EQ(model->subgraphs()->size(), 2);
  EXPECT_EQ(model->subgraphs()->Get(1)->name()->str(), "decomposition");
  const tflite::Operator* op = model->subgraphs()->Get(0)->operators()->Get(0);
  EXPECT_EQ(model->operator_codes()->Get(op->opcode_index())->builtin_code(),
            tflite::BuiltinOperator_STABLEHLO_COMPOSITE);
  const tflite::StableHLOCompositeOptions* options =
      op->builtin_options_2_as_StableHLOCompositeOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(options->name()->str(), "odml.identity");
  EXPECT_EQ(options->decomposition_subgraph_index(), 1);
  const flexbuffers::Map attributes =
      flexbuffers::GetRoot(options->composite_attributes()->data(),
                           options->composite_attributes()->size())
          .AsMap();
  EXPECT_EQ(attributes["n"].AsInt64(), 3);
  EXPECT_EQ(attributes["name"].AsString().str(), "id");
}

TEST_F(ExportTest, NamesFoldedCastsUniquely) {
  const std::string file = ExportOrDie(R"mlir(
    func.func @main(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf16>) {
      %cst = "tfl.pseudo_const"() {value = dense<[1.0, 2.0]> : tensor<2xf16>} : () -> tensor<2xf16> loc("weight")
      %cast = "tfl.cast"(%cst) : (tensor<2xf16>) -> tensor<2xf32>
      %0 = "tfl.add"(%arg0, %cast) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
      func.return %0, %cst : tensor<2xf32>, tensor<2xf16>
    }
  )mlir");
  const tflite::Model* model = tflite::GetModel(file.data());

  const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
  ASSERT_EQ(subgraph->tensors()->size(), 4);
  const tflite::Tensor* f16_weight = subgraph->tensors()->Get(1);
  const tflite::Tensor* f32_weight = subgraph->tensors()->Get(2);
  EXPECT_EQ(f16_weight->name()->str(), "weight");
  EXPECT_EQ(f32_weight->name()->str(), "weight1");
  EXPECT_THAT(GetValues<float>(GetData(file, model, f32_weight)),
              ElementsAre(1.0f, 2.0f));
}

TEST_F(ExportTest, ExportsPerAxisQuantization) {
  const std::string file = ExportOrDie(R"mlir(
    func.func @main() -> tensor<2x2x!quant.uniform<i8:f32:0, {0.5:1, 0.25:-1}>> {
      %w = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<i8:f32:0, {0.5:1, 0.25:-1}>>, value = dense<1> : tensor<2x2xi8>} : () -> tensor<2x2x!quant.uniform<i8:f32:0, {0.5:1, 0.25:-1}>>
      func.return %w : tensor<2x2x!quant.uniform<i8:f32:0, {0.5:1, 0.25:-1}>>
    }
  )mlir");
  const tflite::Model* model = tflite::GetModel(file.data());

  const tflite::Tensor* tensor = model->subgraphs()->Get(0)->tensors()->Get(0);
  EXPECT_EQ(tensor->type(), tflite::TensorType_INT8);
  const tflite::QuantizationParameters* quantization = tensor->quantization();
  EXPECT_THAT(std::vector<float>(quantization->scale()->begin(),
                                 quantization->scale()->end()),
              ElementsAre(0.5f, 0.25f));
  EXPECT_THAT(std::vector<int64_t>(quantization->zero_point()->begin(),
                                   quantization->zero_point()->end()),
              ElementsAre(1, -1));
  EXPECT_EQ(quantization->quantized_dimension(), 0);
  EXPECT_THAT(GetValues<int8_t>(GetData(file, model, tensor)),
              ElementsAre(1, 1, 1, 1));
}

TEST_F(ExportTest, ExportsCustomCallsAndSkipsDeclarations) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    func.func private @declaration(tensor<2xf32>) -> tensor<2xf32>
    func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
      %0 = stablehlo.custom_call @my_target(%x) {backend_config = "config"} : (tensor<2xf32>) -> tensor<2xf32>
      return %0 : tensor<2xf32>
    }
  )mlir");
  PassManager pm(&context_);
  pm.addPass(odml::createLegalizeStablehloToVhloPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  std::string file;
  ASSERT_TRUE(Export(*module, file).ok());
  const tflite::Model* model = tflite::GetModel(file.data());

  ASSERT_EQ(model->subgraphs()->size(), 1);
  const tflite::Operator* op = model->subgraphs()->Get(0)->operators()->Get(0);
  EXPECT_EQ(model->operator_codes()->Get(op->opcode_index())->builtin_code(),
            tflite::BuiltinOperator_STABLEHLO_CUSTOM_CALL);
  const tflite::StablehloCustomCallOptions* options =
      op->builtin_options_2_as_StablehloCustomCallOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(options->call_target_name()->str(), "my_target");
  EXPECT_EQ(options->backend_config()->str(), "config");
}

TEST_F(ExportTest, ReportsVhloOpsWithoutKernels) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    func.func @main(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
      %0 = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
      return %0 : tensor<3x2xf32>
    }
  )mlir");
  PassManager pm(&context_);
  pm.addPass(odml::createLegalizeStablehloToVhloPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  std::string diagnostics;
  ScopedDiagnosticHandler handler(&context_, [&](Diagnostic& diagnostic) {
    diagnostics += diagnostic.str();
    return success();
  });
  std::string file;
  EXPECT_FALSE(Export(*module, file).ok());
  EXPECT_THAT(diagnostics,
              HasSubstr("'vhlo.transpose_v1' op is not supported"));
}

TEST_F(ExportTest, ReportsUnsupportedOps) {
  OwningOpRef<ModuleOp> module = Parse(R"mlir(
    func.func private @callee(%x: tensor<2xf32>) -> tensor<2xf32> {
      return %x : tensor<2xf32>
    }
    func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
      %0 = func.call @callee(%x) : (tensor<2xf32>) -> tensor<2xf32>
      return %0 : tensor<2xf32>
    }
  )mlir");
  std::string diagnostics;
  ScopedDiagnosticHandler handler(&context_, [&](Diagnostic& diagnostic) {
    diagnostics += diagnostic.str();
    return success();
  });
  std::string file;
  EXPECT_FALSE(Export(*module, file).ok());
  EXPECT_THAT(diagnostics, HasSubstr("'func.call' op is not supported"));
}

}  // namespace
}  // namespace mlir::TFL
