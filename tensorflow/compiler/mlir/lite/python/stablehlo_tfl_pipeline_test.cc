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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace mlir::TFL {
namespace {

using ::testing::Contains;
using ::testing::Each;
using ::testing::HasSubstr;
using ::testing::Not;
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
        model->subgraphs()->size() == 0) {
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
      func.func @main(
          %x: tensor<2xf32>, %y: tensor<2xf32>, %z: tensor<2xf32>)
          -> tensor<2xf32> {
        %0 = stablehlo.add %x, %y : tensor<2xf32>
            loc("Module.layer_0"("aten_add"("model.py":12:0)))
        %1 = stablehlo.multiply %0, %z : tensor<2xf32>
            loc("jit(decode)/blocks/0/mlp/mul"(
                callsite("fn_inner"("a.py":1:1) at "fn_outer"("b.py":2:1))))
        %2 = stablehlo.subtract %1, %z : tensor<2xf32>
            loc("jit(decode)/blocks/0/mlp/sub"("a.py":3:1))
        return %2 : tensor<2xf32>
      }
    }
  )mlir";

  std::vector<std::string> names = ConvertAndGetTensorNames(kMlir);
  EXPECT_THAT(names, Contains("Module.layer_0"));
  EXPECT_THAT(names, Contains("jit(decode)/blocks/0/mlp/mul"));
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
    if (absl::string_view(name).find("blocks/1/") != absl::string_view::npos) {
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

}  // namespace
}  // namespace mlir::TFL
