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

#include "tensorflow/compiler/mlir/lite/python/stablehlo_tfl_pipeline.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"

namespace mlir::TFL {
namespace {

using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class StableHloTflPipelineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect,
                    mlir::TFL::TensorFlowLiteDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  std::string ConvertToFlatbuffer(absl::string_view mlir_text) {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &context_);
    EXPECT_TRUE(static_cast<bool>(module));
    if (!module) {
      return {};
    }

    tflite::ConverterFlags converter_flags;
    mlir::TFL::QuantizationSpecs quant_specs;
    mlir::TFL::PassConfig pass_config(quant_specs);

    llvm::SmallString<4096> buffer;
    llvm::raw_svector_ostream export_stream(buffer);
    absl::Status status = ConvertStableHloToTFLite(*module, converter_flags,
                                                   pass_config, export_stream);
    EXPECT_TRUE(status.ok()) << status;
    if (!status.ok()) {
      return {};
    }
    return std::string(buffer.str());
  }

  std::vector<std::string> ConvertAndGetTensorNames(
      absl::string_view mlir_text) {
    std::string buffer = ConvertToFlatbuffer(mlir_text);
    if (buffer.empty()) {
      return {};
    }

    const tflite::Model* model = tflite::GetModel(buffer.data());
    EXPECT_NE(model, nullptr);
    if (model == nullptr || model->subgraphs() == nullptr ||
        model->subgraphs()->empty()) {
      return {};
    }

    std::vector<std::string> names;
    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    if (subgraph->tensors() == nullptr) {
      return names;
    }
    for (const tflite::Tensor* tensor : *subgraph->tensors()) {
      if (tensor != nullptr && tensor->name() != nullptr) {
        names.push_back(tensor->name()->str());
      }
    }
    return names;
  }

  mlir::MLIRContext context_;
};

TEST_F(StableHloTflPipelineTest, FlattensInlinedWhereWithoutJitWhereSubscope) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func private @_where(
          %p: tensor<2xi1>, %x: tensor<2xf32>, %y: tensor<2xf32>)
          -> tensor<2xf32> {
        %0 = stablehlo.select %p, %x, %y : tensor<2xi1>, tensor<2xf32>
            loc("select_n"("_where"("lax_numpy.py":10:3)))
        return %0 : tensor<2xf32>
      }

      func.func @main(
          %p: tensor<2xi1>, %x: tensor<2xf32>, %y: tensor<2xf32>)
          -> tensor<2xf32> {
        %0 = func.call @_where(%p, %x, %y)
            : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
            loc("pjit:"("jit(decode)/blocks/0/attn/jit(_where)"(
                "attend"("attn.py":20:5))))
        %1 = func.call @_where(%p, %0, %y)
            : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
            loc("pjit:"("jit(decode)/blocks/0/attn/jit(_where)"(
                "attend"("attn.py":30:5))))
        %2 = stablehlo.add %1, %y : tensor<2xf32>
            loc("jit(decode)/blocks/0/attn/add"("attend"("attn.py":31:5)))
        return %2 : tensor<2xf32>
      }
    }
  )mlir";

  std::vector<std::string> names = ConvertAndGetTensorNames(kMlir);
  EXPECT_THAT(names, Contains("jit(decode)/blocks/0/attn/select_n"));
  EXPECT_THAT(names, Contains("jit(decode)/blocks/0/attn/select_n1"));
  EXPECT_THAT(names, Each(Not(HasSubstr("jit(_where)"))));
}

TEST_F(StableHloTflPipelineTest, PreservesCallerScopeForInlinedJitFunction) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func private @silu(%x: tensor<2xf32>) -> tensor<2xf32> {
        %0 = stablehlo.logistic %x : tensor<2xf32>
            loc("logistic"("silu"("nn.py":5:3)))
        %1 = stablehlo.multiply %x, %0 : tensor<2xf32>
            loc("mul"("silu"("nn.py":6:3)))
        return %1 : tensor<2xf32>
      }

      func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
        %0 = func.call @silu(%x) : (tensor<2xf32>) -> tensor<2xf32>
            loc("pjit:"("jit(decode)/blocks/0/mlp/jit(silu)"(
                "mlp"("mlp.py":15:5))))
        %1 = stablehlo.add %0, %y : tensor<2xf32>
            loc("jit(decode)/blocks/0/mlp/add"("mlp"("mlp.py":16:5)))
        return %1 : tensor<2xf32>
      }
    }
  )mlir";

  std::vector<std::string> names = ConvertAndGetTensorNames(kMlir);
  EXPECT_THAT(names, Contains("jit(decode)/blocks/0/mlp/jit(silu)/logistic"));
  EXPECT_THAT(names, Contains("jit(decode)/blocks/0/mlp/jit(silu)/mul"));
}

TEST_F(StableHloTflPipelineTest, UnwrapsOpTypeAndEnrichesCompositeName) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func private @bmm_decomp(
          %lhs: tensor<1x2x4xf32>, %rhs: tensor<1x4x3xf32>)
          -> tensor<1x2x3xf32> {
        %0 = stablehlo.dot_general %lhs, %rhs,
            batching_dims = [0] x [0], contracting_dims = [2] x [1]
            : (tensor<1x2x4xf32>, tensor<1x4x3xf32>) -> tensor<1x2x3xf32>
        return %0 : tensor<1x2x3xf32>
      }

      func.func @main(
          %lhs: tensor<1x2x4xf32>, %rhs: tensor<1x4x3xf32>,
          %bias: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
        %0 = stablehlo.composite "odml.runtime_bmm" %lhs, %rhs
            {decomposition = @bmm_decomp}
            : (tensor<1x2x4xf32>, tensor<1x4x3xf32>) -> tensor<1x2x3xf32>
            loc("composite:"("jit(decode)/blocks/0/attn/composite"(
                "runtime_bmm"("bmm.py":42:3))))
        %1 = stablehlo.add %0, %bias : tensor<1x2x3xf32>
            loc("jit(decode)/blocks/0/attn/add"("runtime_bmm"("bmm.py":43:3)))
        return %1 : tensor<1x2x3xf32>
      }
    }
  )mlir";

  std::vector<std::string> names = ConvertAndGetTensorNames(kMlir);
  EXPECT_THAT(names, Contains("jit(decode)/blocks/0/attn/odml.runtime_bmm"));
  EXPECT_THAT(names, Each(Not(HasSubstr("vhlo.composite_v1"))));
}

TEST_F(StableHloTflPipelineTest,
       PreservesPytorchNestedNameLocAndTracebackCallSite) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func private @_linear_pointwise(
          %x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
        %0 = stablehlo.add %x, %y : tensor<2xf32> loc(unknown)
        return %0 : tensor<2xf32>
      }

      func.func @main(
          %x: tensor<2xf32>, %y: tensor<2xf32>, %z: tensor<2xf32>)
          -> tensor<2xf32> {
        %0 = stablehlo.add %x, %y : tensor<2xf32>
            loc("Module.layer_0"("aten_add"("model.py":12:0)))
        %1 = stablehlo.multiply %0, %z : tensor<2xf32>
            loc("jit(decode)/blocks/0/mlp/mul"(
                callsite("fn_inner"("a.py":1:1) at "fn_outer"("b.py":2:1))))
        %2 = func.call @_linear_pointwise(%1, %z)
            : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
            loc("torch.nn.modules.linear.Linear_fc;/model/fc;Linear"(
                "model.py":13:0))
        %3 = stablehlo.subtract %2, %z : tensor<2xf32>
            loc("jit(decode)/blocks/0/mlp/sub"("a.py":3:1))
        return %3 : tensor<2xf32>
      }
    }
  )mlir";

  std::vector<std::string> names = ConvertAndGetTensorNames(kMlir);
  EXPECT_THAT(names, Contains("Module.layer_0"));
  EXPECT_THAT(names, Contains("jit(decode)/blocks/0/mlp/mul"));
  EXPECT_THAT(names,
              Contains("torch.nn.modules.linear.Linear_fc;/model/fc;Linear"));
  EXPECT_THAT(names, Each(Not(HasSubstr("aten_add"))));
  EXPECT_THAT(names, Each(Not(HasSubstr("fn_outer"))));
}

TEST_F(StableHloTflPipelineTest, DropsCrossScopeCseOutlierFromFusedLocations) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func @main(
          %p: tensor<4xi1>, %x0: tensor<2x4xf32>, %x1: tensor<2x4xf32>,
          %y: tensor<2x4xf32>) -> tensor<2x4xf32> {
        %b0 = stablehlo.broadcast_in_dim %p, dims = [1]
            : (tensor<4xi1>) -> tensor<2x4xi1>
            loc("jit(decode)/blocks/0/attn/broadcast_in_dim"("a.py":1:1))
        %s0 = stablehlo.select %b0, %x0, %y : tensor<2x4xi1>, tensor<2x4xf32>
            loc("jit(decode)/blocks/0/attn/select_n"("a.py":2:1))
        %b1 = stablehlo.broadcast_in_dim %p, dims = [1]
            : (tensor<4xi1>) -> tensor<2x4xi1>
            loc("jit(decode)/blocks/1/attn/broadcast_in_dim"("a.py":3:1))
        %s1 = stablehlo.select %b1, %x1, %y : tensor<2x4xi1>, tensor<2x4xf32>
            loc("jit(decode)/blocks/1/attn/select_n"("a.py":4:1))
        %sm1 = "tfl.softmax"(%s1) {beta = 1.0 : f32}
            : (tensor<2x4xf32>) -> tensor<2x4xf32>
            loc(fused[
                "jit(decode)/blocks/1/attn/div"("a.py":5:1),
                "jit(decode)/blocks/1/attn/exp"("a.py":6:1),
                "jit(decode)/blocks/0/attn/max"("a.py":7:1),
                "jit(decode)/blocks/1/attn/sub"("a.py":8:1)])
        %out = stablehlo.add %s0, %sm1 : tensor<2x4xf32>
            loc("jit(decode)/out/add"("a.py":9:1))
        return %out : tensor<2x4xf32>
      }
    }
  )mlir";

  std::vector<std::string> names = ConvertAndGetTensorNames(kMlir);
  EXPECT_THAT(names, Contains("jit(decode)/blocks/1/attn/select_n"));
  EXPECT_THAT(names, Contains("jit(decode)/blocks/1/attn/div;"
                              "jit(decode)/blocks/1/attn/exp;"
                              "jit(decode)/blocks/1/attn/sub"));
  for (const std::string& name : names) {
    if (absl::StrContains(name, "blocks/1/")) {
      EXPECT_THAT(name, Not(HasSubstr("blocks/0/")));
    }
  }
}

TEST_F(StableHloTflPipelineTest,
       PreservesHierarchicalScopeOnEntryFunctionOutputs) {
  constexpr absl::string_view kMlir = R"mlir(
    module attributes {tf_saved_model.semantics} {
      func.func private @cache_update_decomp(
          %k: tensor<1x4xf32>, %v: tensor<1x4xf32>, %u: tensor<1x4xf32>)
          -> (tensor<1x4xf32>, tensor<1x4xf32>) {
        %0 = stablehlo.add %k, %u : tensor<1x4xf32>
        %1 = stablehlo.add %v, %u : tensor<1x4xf32>
        return %0, %1 : tensor<1x4xf32>, tensor<1x4xf32>
      }

      func.func @decode(
          %k: tensor<1x4xf32> {tf_saved_model.index_path = ["kv_cache_k_0"]},
          %v: tensor<1x4xf32> {tf_saved_model.index_path = ["kv_cache_v_0"]},
          %x: tensor<1x4xf32> {tf_saved_model.index_path = ["embeddings"]})
          -> (tensor<1x4xf32> {tf_saved_model.index_path = ["kv_cache_k_0"]},
              tensor<1x4xf32> {tf_saved_model.index_path = ["kv_cache_v_0"]},
              tensor<1x4xf32> {tf_saved_model.index_path = ["logits"]})
          attributes {
            tf.entry_function = {
              inputs = "decode_kv_cache_k_0,decode_kv_cache_v_0,decode_embeddings",
              outputs = "decode_kv_cache_k_0_output,decode_kv_cache_v_0_output,decode_logits_output"
            },
            tf_saved_model.exported_names = ["decode"]
          } {
        %0:2 = stablehlo.composite "odml.cache_update" %k, %v, %x
            {decomposition = @cache_update_decomp}
            : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>)
            -> (tensor<1x4xf32>, tensor<1x4xf32>)
            loc("composite:"("jit(decode)/blocks/0/attn/update/composite"(
                "update"("cache.py":10:3))))
        %1 = stablehlo.add %0#0, %x : tensor<1x4xf32>
            loc("jit(decode)/lm_head/bt,vh->bv/add"("head.py":20:3))
        return %0#0, %0#1, %1
            : tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>
      }
    }
  )mlir";

  std::string buffer = ConvertToFlatbuffer(kMlir);
  ASSERT_FALSE(buffer.empty());
  const tflite::Model* model = tflite::GetModel(buffer.data());
  ASSERT_NE(model, nullptr);
  ASSERT_NE(model->subgraphs(), nullptr);
  ASSERT_GE(model->subgraphs()->size(), 1u);

  const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
  std::vector<std::string> names;
  for (const tflite::Tensor* tensor : *subgraph->tensors()) {
    if (tensor != nullptr && tensor->name() != nullptr) {
      names.push_back(tensor->name()->str());
    }
  }
  EXPECT_THAT(names,
              Contains("jit(decode)/blocks/0/attn/update/odml.cache_update"));
  EXPECT_THAT(names,
              Contains("jit(decode)/blocks/0/attn/update/odml.cache_update1"));
  EXPECT_THAT(names, Contains("jit(decode)/lm_head/bt_vh->bv/add"));
  EXPECT_THAT(names, Each(Not(HasSubstr("_output"))));

  ASSERT_NE(model->signature_defs(), nullptr);
  ASSERT_EQ(model->signature_defs()->size(), 1u);
  const tflite::SignatureDef* sig = model->signature_defs()->Get(0);
  EXPECT_EQ(sig->signature_key()->str(), "decode");
  std::vector<std::string> sig_output_names;
  for (const tflite::TensorMap* out_map : *sig->outputs()) {
    sig_output_names.push_back(out_map->name()->str());
  }
  EXPECT_THAT(sig_output_names,
              UnorderedElementsAre("kv_cache_k_0", "kv_cache_v_0", "logits"));
}

TEST_F(StableHloTflPipelineTest, LegalizesUnfusedTrainingBatchNorm) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func @main(%x: tensor<2x3xf32>, %scale: tensor<3xf32>,
                      %offset: tensor<3xf32>) -> tensor<2x3xf32> {
        %0:3 = "stablehlo.batch_norm_training"(%x, %scale, %offset)
            {epsilon = 1.0e-3 : f32, feature_index = 1 : i64}
            : (tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>)
            -> (tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>)
        return %0#0 : tensor<2x3xf32>
      }
    }
  )mlir";

  std::string buffer = ConvertToFlatbuffer(kMlir);
  ASSERT_FALSE(buffer.empty());
  const tflite::Model* model = tflite::GetModel(buffer.data());
  bool has_mean = false;
  for (const tflite::OperatorCode* opcode : *model->operator_codes()) {
    has_mean |= opcode->builtin_code() == tflite::BuiltinOperator_MEAN;
  }
  EXPECT_TRUE(has_mean);
}

TEST_F(StableHloTflPipelineTest, PreservesIrLocationsWithoutMutation) {
  constexpr absl::string_view kMlir = R"mlir(
    module {
      func.func private @silu(%x: tensor<2xf32>) -> tensor<2xf32> {
        %0 = stablehlo.logistic %x : tensor<2xf32>
            loc("logistic"("silu"("nn.py":5:3)))
        %1 = stablehlo.multiply %x, %0 : tensor<2xf32>
            loc("mul"("silu"("nn.py":6:3)))
        return %1 : tensor<2xf32>
      }

      func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32>
          attributes {
            tf.entry_function = {
              inputs = "main_x,main_y",
              outputs = "main_output"
            }
          } {
        %0 = func.call @silu(%x) : (tensor<2xf32>) -> tensor<2xf32>
            loc("pjit:"("jit(decode)/blocks/0/mlp/jit(silu)"(
                "mlp"("mlp.py":15:5))))
        %1 = stablehlo.add %0, %y : tensor<2xf32>
            loc("jit(decode)/blocks/0/mlp/add"("mlp"("mlp.py":16:5)))
        return %1 : tensor<2xf32>
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kMlir, &context_);
  ASSERT_TRUE(static_cast<bool>(module));

  tflite::ConverterFlags converter_flags;
  mlir::TFL::QuantizationSpecs quant_specs;
  mlir::TFL::PassConfig pass_config(quant_specs);
  llvm::SmallString<4096> buffer;
  llvm::raw_svector_ostream export_stream(buffer);
  ASSERT_TRUE(ConvertStableHloToTFLite(*module, converter_flags, pass_config,
                                       export_stream)
                  .ok());

  bool found_callsite_with_pjit_wrapper = false;
  module->walk([&](mlir::Operation* op) {
    if (auto call_loc = mlir::dyn_cast<mlir::CallSiteLoc>(op->getLoc())) {
      if (auto caller_name_loc =
              mlir::dyn_cast<mlir::NameLoc>(call_loc.getCaller())) {
        if (caller_name_loc.getName().strref() == "pjit:") {
          found_callsite_with_pjit_wrapper = true;
        }
      }
    }
  });
  EXPECT_TRUE(found_callsite_with_pjit_wrapper);

  mlir::func::FuncOp main_fn = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(static_cast<bool>(main_fn));
  auto dict_attr =
      main_fn->getAttrOfType<mlir::DictionaryAttr>("tf.entry_function");
  ASSERT_TRUE(static_cast<bool>(dict_attr));
  auto outputs_attr =
      mlir::dyn_cast_or_null<mlir::StringAttr>(dict_attr.get("outputs"));
  ASSERT_TRUE(static_cast<bool>(outputs_attr));
  EXPECT_EQ(outputs_attr.getValue().str(), "main_output");
}

// End-to-end tests for preserve regions (`odml.preserve_region` composites).
class PreserveRegionPipelineTest : public ::testing::Test {
 protected:
  PreserveRegionPipelineTest() {
    context_.loadDialect<func::FuncDialect, stablehlo::StablehloDialect,
                         TFL::TensorFlowLiteDialect>();
  }

  // Converts `ir` and returns the flatbuffer, or an empty buffer on failure.
  llvm::SmallString<0> Convert(llvm::StringRef ir, bool enable_x64 = true) {
    llvm::SmallString<0> buffer;
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context_);
    EXPECT_TRUE(module);
    if (!module) return buffer;

    tflite::ConverterFlags converter_flags;
    converter_flags.set_enable_x64(enable_x64);
    PassConfig pass_config{QuantizationSpecs()};
    llvm::raw_svector_ostream stream(buffer);
    absl::Status status =
        ConvertStableHloToTFLite(*module, converter_flags, pass_config, stream);
    EXPECT_TRUE(status.ok()) << status;
    if (!status.ok()) buffer.clear();
    return buffer;
  }

  // Returns the builtin operators of the main subgraph, in order.
  static std::vector<tflite::BuiltinOperator> Ops(const tflite::Model* model) {
    std::vector<tflite::BuiltinOperator> ops;
    for (const tflite::Operator* op :
         *model->subgraphs()->Get(0)->operators()) {
      ops.push_back(tflite::GetBuiltinCode(
          model->operator_codes()->Get(op->opcode_index())));
    }
    return ops;
  }

  static const tflite::Operator* FindOp(const tflite::Model* model,
                                        tflite::BuiltinOperator code) {
    for (const tflite::Operator* op :
         *model->subgraphs()->Get(0)->operators()) {
      if (tflite::GetBuiltinCode(
              model->operator_codes()->Get(op->opcode_index())) == code) {
        return op;
      }
    }
    return nullptr;
  }

  static const tflite::Tensor* Input(const tflite::Model* model,
                                     const tflite::Operator* op, int index) {
    return model->subgraphs()->Get(0)->tensors()->Get(op->inputs()->Get(index));
  }

  // Returns the contents of `tensor`'s buffer, inline or at an offset.
  static std::vector<uint8_t> BufferData(llvm::StringRef flatbuffer,
                                         const tflite::Model* model,
                                         const tflite::Tensor* tensor) {
    const tflite::Buffer* buffer = model->buffers()->Get(tensor->buffer());
    if (buffer->data() != nullptr && !buffer->data()->empty()) {
      return {buffer->data()->begin(), buffer->data()->end()};
    }
    if (buffer->offset() > 1 &&
        buffer->offset() + buffer->size() <= flatbuffer.size()) {
      const auto* begin = reinterpret_cast<const uint8_t*>(flatbuffer.data()) +
                          buffer->offset();
      return {begin, begin + buffer->size()};
    }
    return {};
  }

  MLIRContext context_;
};

// The same authored `tfl.reshape(tfl.reshape(x))` chain inside and outside a
// preserve region. The TFL optimization pipeline folds the one outside into a
// single reshape; the one inside is kept as authored.
TEST_F(PreserveRegionPipelineTest, AuthoredOpsInPreserveRegionAreKept) {
  llvm::SmallString<0> flatbuffer = Convert(R"mlir(
module {
  func.func @main(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> (tensor<24xf32>, tensor<24xf32>) {
    %s0 = stablehlo.constant dense<[6, 4]> : tensor<2xi32>
    %s1 = stablehlo.constant dense<24> : tensor<1xi32>
    %0 = stablehlo.custom_call @tfl.reshape(%arg0, %s0) {backend_config = "{}"} : (tensor<2x3x4xf32>, tensor<2xi32>) -> tensor<6x4xf32>
    %1 = stablehlo.custom_call @tfl.reshape(%0, %s1) {backend_config = "{}"} : (tensor<6x4xf32>, tensor<1xi32>) -> tensor<24xf32>
    %2 = stablehlo.composite "odml.preserve_region" %arg1, %s0, %s1 {
      decomposition = @region
    } : (tensor<2x3x4xf32>, tensor<2xi32>, tensor<1xi32>) -> tensor<24xf32>
    return %1, %2 : tensor<24xf32>, tensor<24xf32>
  }
  func.func private @region(%x: tensor<2x3x4xf32>, %a: tensor<2xi32>, %b: tensor<1xi32>) -> tensor<24xf32> {
    %0 = stablehlo.custom_call @tfl.reshape(%x, %a) {backend_config = "{}"} : (tensor<2x3x4xf32>, tensor<2xi32>) -> tensor<6x4xf32>
    %1 = stablehlo.custom_call @tfl.reshape(%0, %b) {backend_config = "{}"} : (tensor<6x4xf32>, tensor<1xi32>) -> tensor<24xf32>
    return %1 : tensor<24xf32>
  }
}
)mlir");
  ASSERT_FALSE(flatbuffer.empty());
  const tflite::Model* model = tflite::GetModel(flatbuffer.data());
  ASSERT_EQ(model->subgraphs()->size(), 1);
  EXPECT_THAT(Ops(model), SizeIs(3));
  EXPECT_THAT(Ops(model), Each(Eq(tflite::BuiltinOperator_RESHAPE)));
}

// A resource-backed i64 constant used only inside a region, with
// `enable_x64=false`. Both pipelines narrow 64-bit types, so the boundary
// still matches, and the narrowed constant, which only the region body
// references while the main module is optimized, must keep its data.
TEST_F(PreserveRegionPipelineTest, RegionResourceConstantsSurvive) {
  llvm::SmallString<0> flatbuffer = Convert(R"mlir(
module {
  func.func @main(%arg0: tensor<4xi64>) -> tensor<4xi64> {
    %w = stablehlo.constant dense_resource<region_weights> : tensor<4xi64>
    %0 = stablehlo.composite "odml.preserve_region" %arg0, %w {
      decomposition = @region
    } : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
  }
  func.func private @region(%x: tensor<4xi64>, %w: tensor<4xi64>) -> tensor<4xi64> {
    %0 = stablehlo.custom_call @tfl.add(%x, %w) {backend_config = "{fused_activation_function = \"NONE\"}"} : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
  }
}
{-#
  dialect_resources: {
    builtin: {
      region_weights: "0x080000000100000000000000020000000000000003000000000000000400000000000000"
    }
  }
#-}
)mlir",
                                            /*enable_x64=*/false);
  ASSERT_FALSE(flatbuffer.empty());
  const tflite::Model* model = tflite::GetModel(flatbuffer.data());
  EXPECT_THAT(Ops(model), ElementsAre(tflite::BuiltinOperator_ADD));

  const tflite::Operator* add = FindOp(model, tflite::BuiltinOperator_ADD);
  ASSERT_NE(add, nullptr);
  const tflite::Tensor* weights = Input(model, add, 1);
  EXPECT_EQ(weights->type(), tflite::TensorType_INT32);
  std::vector<uint8_t> data = BufferData(flatbuffer, model, weights);
  ASSERT_EQ(data.size(), 4 * sizeof(int32_t));
  int32_t values[4];
  std::memcpy(values, data.data(), sizeof(values));
  EXPECT_THAT(values, ElementsAre(1, 2, 3, 4));
}

// Authored Q/DQ ops inside a region still fuse into a quantized kernel: the
// region pipeline runs quantization even though it skips optimization.
TEST_F(PreserveRegionPipelineTest, AuthoredQdqInPreserveRegionIsFused) {
  llvm::SmallString<0> flatbuffer = Convert(R"mlir(
module {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<24xf32> {
    %s = stablehlo.constant dense<24> : tensor<1xi32>
    %0 = stablehlo.composite "odml.preserve_region" %arg0, %s {
      decomposition = @region
    } : (tensor<2x3x4xf32>, tensor<1xi32>) -> tensor<24xf32>
    return %0 : tensor<24xf32>
  }
  func.func private @region(%x: tensor<2x3x4xf32>, %s: tensor<1xi32>) -> tensor<24xf32> {
    %q = stablehlo.custom_call @tfl.quantize(%x) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    %dq = stablehlo.custom_call @tfl.dequantize(%q) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    %r = stablehlo.custom_call @tfl.reshape(%dq, %s) {backend_config = "{}"} : (tensor<2x3x4xf32>, tensor<1xi32>) -> tensor<24xf32>
    %q2 = stablehlo.custom_call @tfl.quantize(%r) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<24xf32>) -> tensor<24xf32>
    %dq2 = stablehlo.custom_call @tfl.dequantize(%q2) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<24xf32>) -> tensor<24xf32>
    return %dq2 : tensor<24xf32>
  }
}
)mlir");
  ASSERT_FALSE(flatbuffer.empty());
  const tflite::Model* model = tflite::GetModel(flatbuffer.data());
  EXPECT_THAT(Ops(model), ElementsAre(tflite::BuiltinOperator_QUANTIZE,
                                      tflite::BuiltinOperator_RESHAPE,
                                      tflite::BuiltinOperator_DEQUANTIZE));
  const tflite::Operator* reshape =
      FindOp(model, tflite::BuiltinOperator_RESHAPE);
  ASSERT_NE(reshape, nullptr);
  EXPECT_EQ(Input(model, reshape, 0)->type(), tflite::TensorType_INT8);
}

// The main pipeline propagates quantization parameters, so a float reshape
// fed by a dequantize gets quantized there. Inside a region the reshape stays
// float, as authored: the region pipeline does not run PropagateQParams.
constexpr char kFloatReshapeAfterDq[] = R"mlir(
    %q = stablehlo.custom_call @tfl.quantize(%x) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    %dq = stablehlo.custom_call @tfl.dequantize(%q) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    %r = stablehlo.custom_call @tfl.reshape(%dq, %s) {backend_config = "{}"} : (tensor<2x3x4xf32>, tensor<1xi32>) -> tensor<24xf32>
)mlir";

TEST_F(PreserveRegionPipelineTest,
       FloatReshapeAfterDqIsQuantizedOutsideRegion) {
  llvm::SmallString<0> flatbuffer = Convert(absl::StrCat(R"mlir(
module {
  func.func @main(%x: tensor<2x3x4xf32>) -> tensor<24xf32> {
    %s = stablehlo.constant dense<24> : tensor<1xi32>
)mlir",
                                                         kFloatReshapeAfterDq,
                                                         R"mlir(
    return %r : tensor<24xf32>
  }
}
)mlir"));
  ASSERT_FALSE(flatbuffer.empty());
  const tflite::Model* model = tflite::GetModel(flatbuffer.data());
  const tflite::Operator* reshape =
      FindOp(model, tflite::BuiltinOperator_RESHAPE);
  ASSERT_NE(reshape, nullptr);
  EXPECT_EQ(Input(model, reshape, 0)->type(), tflite::TensorType_INT8);
}

TEST_F(PreserveRegionPipelineTest,
       FloatReshapeAfterDqStaysFloatInPreserveRegion) {
  llvm::SmallString<0> flatbuffer = Convert(absl::StrCat(R"mlir(
module {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<24xf32> {
    %s = stablehlo.constant dense<24> : tensor<1xi32>
    %0 = stablehlo.composite "odml.preserve_region" %arg0, %s {
      decomposition = @region
    } : (tensor<2x3x4xf32>, tensor<1xi32>) -> tensor<24xf32>
    return %0 : tensor<24xf32>
  }
  func.func private @region(%x: tensor<2x3x4xf32>, %s: tensor<1xi32>) -> tensor<24xf32> {
)mlir",
                                                         kFloatReshapeAfterDq,
                                                         R"mlir(
    return %r : tensor<24xf32>
  }
}
)mlir"));
  ASSERT_FALSE(flatbuffer.empty());
  const tflite::Model* model = tflite::GetModel(flatbuffer.data());
  EXPECT_THAT(Ops(model), ElementsAre(tflite::BuiltinOperator_QUANTIZE,
                                      tflite::BuiltinOperator_DEQUANTIZE,
                                      tflite::BuiltinOperator_RESHAPE));
  const tflite::Operator* reshape =
      FindOp(model, tflite::BuiltinOperator_RESHAPE);
  ASSERT_NE(reshape, nullptr);
  EXPECT_EQ(Input(model, reshape, 0)->type(), tflite::TensorType_FLOAT32);
}

// An op whose input and weights the author quantized gets an int32 bias in a
// region, so it still fuses into a valid quantized kernel.
TEST_F(PreserveRegionPipelineTest,
       AuthoredQuantizedFcInPreserveRegionGetsInt32Bias) {
  llvm::SmallString<0> flatbuffer = Convert(R"mlir(
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x2xf32> {
    %0 = stablehlo.composite "odml.preserve_region" %arg0 {
      decomposition = @region
    } : (tensor<1x4xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
  func.func private @region(%x: tensor<1x4xf32>) -> tensor<1x2xf32> {
    %w = stablehlo.constant dense<0.5> : tensor<2x4xf32>
    %b = stablehlo.constant dense<1.0> : tensor<2xf32>
    %qx = stablehlo.custom_call @tfl.quantize(%x) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %dqx = stablehlo.custom_call @tfl.dequantize(%qx) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %qw = stablehlo.custom_call @tfl.quantize(%w) {backend_config = "{scale = 0.01 : f32, zero_point = 0 : i64}"} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %dqw = stablehlo.custom_call @tfl.dequantize(%qw) {backend_config = "{scale = 0.01 : f32, zero_point = 0 : i64}"} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %fc = stablehlo.custom_call @tfl.fully_connected(%dqx, %dqw, %b) {backend_config = "{fused_activation_function = \"NONE\", keep_num_dims = false, weights_format = \"DEFAULT\"}"} : (tensor<1x4xf32>, tensor<2x4xf32>, tensor<2xf32>) -> tensor<1x2xf32>
    %qy = stablehlo.custom_call @tfl.quantize(%fc) {backend_config = "{scale = 0.2 : f32, zero_point = 0 : i64}"} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %dqy = stablehlo.custom_call @tfl.dequantize(%qy) {backend_config = "{scale = 0.2 : f32, zero_point = 0 : i64}"} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %dqy : tensor<1x2xf32>
  }
}
)mlir");
  ASSERT_FALSE(flatbuffer.empty());
  const tflite::Model* model = tflite::GetModel(flatbuffer.data());
  const tflite::Operator* fc =
      FindOp(model, tflite::BuiltinOperator_FULLY_CONNECTED);
  ASSERT_NE(fc, nullptr);
  EXPECT_EQ(Input(model, fc, 0)->type(), tflite::TensorType_INT8);
  EXPECT_EQ(Input(model, fc, 1)->type(), tflite::TensorType_INT8);
  EXPECT_EQ(Input(model, fc, 2)->type(), tflite::TensorType_INT32);
}

// A quantized value cannot cross the region boundary: legalizing the quantize
// in main gives the composite operand a quantized type that no longer matches
// the float signature of its decomposition, which fails verification. Q/DQ
// pairs must be authored on one side of the boundary.
TEST_F(PreserveRegionPipelineTest, QuantizedValueCrossingRegionBoundaryFails) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<24xf32> {
    %s = stablehlo.constant dense<24> : tensor<1xi32>
    %q = stablehlo.custom_call @tfl.quantize(%arg0) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    %0 = stablehlo.composite "odml.preserve_region" %q, %s {
      decomposition = @region
    } : (tensor<2x3x4xf32>, tensor<1xi32>) -> tensor<24xf32>
    return %0 : tensor<24xf32>
  }
  func.func private @region(%x: tensor<2x3x4xf32>, %s: tensor<1xi32>) -> tensor<24xf32> {
    %dq = stablehlo.custom_call @tfl.dequantize(%x) {backend_config = "{scale = 0.1 : f32, zero_point = 0 : i64}"} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    %r = stablehlo.custom_call @tfl.reshape(%dq, %s) {backend_config = "{}"} : (tensor<2x3x4xf32>, tensor<1xi32>) -> tensor<24xf32>
    return %r : tensor<24xf32>
  }
}
)mlir",
                                                             &context_);
  ASSERT_TRUE(module);
  tflite::ConverterFlags converter_flags;
  PassConfig pass_config{QuantizationSpecs()};
  llvm::SmallString<0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  absl::Status status =
      ConvertStableHloToTFLite(*module, converter_flags, pass_config, stream);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), HasSubstr("but decomposition has type"));
}

// bf16 at the region boundary: both pipelines leave the composite and function
// signatures in bf16 and legalize inside, so stitching succeeds and the region
// computes in f32.
TEST_F(PreserveRegionPipelineTest, Bf16PreserveRegionBoundary) {
  llvm::SmallString<0> flatbuffer = Convert(R"mlir(
module {
  func.func @main(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {
    %0 = stablehlo.composite "odml.preserve_region" %arg0 {
      decomposition = @region
    } : (tensor<4xbf16>) -> tensor<4xbf16>
    return %0 : tensor<4xbf16>
  }
  func.func private @region(%x: tensor<4xbf16>) -> tensor<4xbf16> {
    %0 = stablehlo.custom_call @tfl.add(%x, %x) {backend_config = "{fused_activation_function = \"NONE\"}"} : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xbf16>
    return %0 : tensor<4xbf16>
  }
}
)mlir");
  ASSERT_FALSE(flatbuffer.empty());
  const tflite::Model* model = tflite::GetModel(flatbuffer.data());
  const tflite::Operator* add = FindOp(model, tflite::BuiltinOperator_ADD);
  ASSERT_NE(add, nullptr);
  EXPECT_EQ(Input(model, add, 0)->type(), tflite::TensorType_FLOAT32);
}

// Pins what preserve regions do not protect against: the quantization passes
// in the region pipeline run op folders, so an identity reshape still folds.
// The non-identity reshape is kept.
TEST_F(PreserveRegionPipelineTest, IdentityOpsInPreserveRegionStillFold) {
  llvm::SmallString<0> flatbuffer = Convert(R"mlir(
module {
  func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<24xf32> {
    %s0 = stablehlo.constant dense<[2, 3, 4]> : tensor<3xi32>
    %s1 = stablehlo.constant dense<24> : tensor<1xi32>
    %0 = stablehlo.composite "odml.preserve_region" %arg0, %s0, %s1 {
      decomposition = @region
    } : (tensor<2x3x4xf32>, tensor<3xi32>, tensor<1xi32>) -> tensor<24xf32>
    return %0 : tensor<24xf32>
  }
  func.func private @region(%x: tensor<2x3x4xf32>, %a: tensor<3xi32>, %b: tensor<1xi32>) -> tensor<24xf32> {
    %0 = stablehlo.custom_call @tfl.reshape(%x, %a) {backend_config = "{}"} : (tensor<2x3x4xf32>, tensor<3xi32>) -> tensor<2x3x4xf32>
    %1 = stablehlo.custom_call @tfl.reshape(%0, %b) {backend_config = "{}"} : (tensor<2x3x4xf32>, tensor<1xi32>) -> tensor<24xf32>
    return %1 : tensor<24xf32>
  }
}
)mlir");
  ASSERT_FALSE(flatbuffer.empty());
  const tflite::Model* model = tflite::GetModel(flatbuffer.data());
  EXPECT_THAT(Ops(model), ElementsAre(tflite::BuiltinOperator_RESHAPE));
}

}  // namespace
}  // namespace mlir::TFL
