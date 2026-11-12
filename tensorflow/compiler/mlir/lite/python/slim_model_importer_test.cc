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

#include "tensorflow/compiler/mlir/lite/python/slim_model_importer.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "xla/tsl/platform/env.h"

namespace tensorflow {
namespace {

std::string JoinPath(const std::string& base, const std::string& child) {
  return base + "/" + child;
}

void WriteFile(const std::string& path, const std::string& content) {
  std::ofstream out(path, std::ios::binary);
  out.write(content.data(), content.size());
}

class SlimModelImporterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry
        .insert<mlir::func::FuncDialect, mlir::TFL::TensorFlowLiteDialect,
                mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::MLIRContext context_;
};

TEST_F(SlimModelImporterTest, MissingMetadataFile) {
  std::string non_existent_dir =
      JoinPath(testing::TempDir(), "non_existent_dir");
  auto result = LoadSlimModel(non_existent_dir, &context_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kNotFound);
}

TEST_F(SlimModelImporterTest, InvalidJsonMetadata) {
  std::string dir = JoinPath(testing::TempDir(), "invalid_json_dir");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  std::string metadata_path = JoinPath(dir, "weights_metadata.json");
  WriteFile(metadata_path, "{ invalid json }");

  auto result = LoadSlimModel(dir, &context_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(SlimModelImporterTest, MissingSignaturesField) {
  std::string dir = JoinPath(testing::TempDir(), "missing_signatures_dir");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  std::string metadata_path = JoinPath(dir, "weights_metadata.json");
  WriteFile(metadata_path, R"({"foo": "bar"})");

  auto result = LoadSlimModel(dir, &context_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(SlimModelImporterTest, MissingMlirbcFile) {
  std::string dir = JoinPath(testing::TempDir(), "missing_mlirbc_dir");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  std::string metadata_path = JoinPath(dir, "weights_metadata.json");
  WriteFile(metadata_path, R"({"signatures": {"serve": []}})");

  auto result = LoadSlimModel(dir, &context_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
}

TEST_F(SlimModelImporterTest, ValidSlimModelImportWithWeightInjection) {
  std::string dir = JoinPath(testing::TempDir(), "valid_slim_model");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  // Write metadata JSON
  std::string metadata_json = R"({
    "signatures": {
      "serve": [
        {"arg_index": 1, "offset": 0}
      ]
    },
    "signature_inputs": {
      "serve": ["x"]
    },
    "signature_outputs": {
      "serve": ["y"]
    }
  })";
  WriteFile(JoinPath(dir, "weights_metadata.json"), metadata_json);

  // Write params.bin (float 3.14f -> 4 bytes)
  float weight_val = 3.14f;
  std::string params_data(reinterpret_cast<const char*>(&weight_val),
                          sizeof(float));
  WriteFile(JoinPath(dir, "params.bin"), params_data);

  // Write serve.mlirbc
  std::string mlir_content = R"(
    module {
      func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
        return %0 : tensor<1xf32>
      }
    }
  )";
  WriteFile(JoinPath(dir, "serve.mlirbc"), mlir_content);

  // Load slim model
  auto result_or = LoadSlimModel(dir, &context_);
  ASSERT_TRUE(result_or.ok());

  mlir::OwningOpRef<mlir::ModuleOp> module = std::move(result_or.value());
  ASSERT_NE(module.get(), nullptr);

  // Check top-level attribute
  EXPECT_TRUE(module.get()->hasAttr("tf_saved_model.semantics"));

  // Check function renaming to @serve
  auto serve_func = module->lookupSymbol<mlir::func::FuncOp>("serve");
  ASSERT_NE(serve_func, nullptr);

  // Argument 1 should have been replaced with constant & erased, leaving 1
  // argument (%arg0)
  EXPECT_EQ(serve_func.getNumArguments(), 1);

  // Check exported names attribute
  auto exported_names = serve_func->getAttrOfType<mlir::ArrayAttr>(
      "tf_saved_model.exported_names");
  ASSERT_NE(exported_names, nullptr);
  EXPECT_EQ(exported_names.size(), 1);
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(exported_names[0]).getValue(),
            "serve");

  // Check arg attribute tf_saved_model.index_path
  auto arg_index_path = serve_func.getArgAttrOfType<mlir::ArrayAttr>(
      0, "tf_saved_model.index_path");
  ASSERT_NE(arg_index_path, nullptr);
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(arg_index_path[0]).getValue(), "x");

  // Check result attribute tf_saved_model.index_path
  auto res_index_path = serve_func.getResultAttrOfType<mlir::ArrayAttr>(
      0, "tf_saved_model.index_path");
  ASSERT_NE(res_index_path, nullptr);
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(res_index_path[0]).getValue(), "y");
}

// Producers such as JAX serialize for older consumers, so a model can hold
// VHLO ops older than the current version. They must be upgraded first.
TEST_F(SlimModelImporterTest, UpgradesOlderVhloOpVersions) {
  std::string dir = JoinPath(testing::TempDir(), "older_vhlo_dir");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());
  WriteFile(JoinPath(dir, "weights_metadata.json"),
            R"({"signatures": {"serve": []}})");

  // `vhlo.dot_general_v1` is only legal up to VHLO 1.5.0.
  std::string mlir_content = R"(
    module {
      vhlo.func_v1 @main(%arg0: !vhlo.tensor_v1<1x!vhlo.f32_v1>, %arg1: !vhlo.tensor_v1<1x!vhlo.f32_v1>) -> (!vhlo.tensor_v1<!vhlo.f32_v1>) {
        %0 = "vhlo.dot_general_v1"(%arg0, %arg1) <{lhs_batching_dimensions = #vhlo.tensor_v1<dense<> : tensor<0xi64>>, lhs_contracting_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>, precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>, rhs_batching_dimensions = #vhlo.tensor_v1<dense<> : tensor<0xi64>>, rhs_contracting_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>}> : (!vhlo.tensor_v1<1x!vhlo.f32_v1>, !vhlo.tensor_v1<1x!vhlo.f32_v1>) -> !vhlo.tensor_v1<!vhlo.f32_v1>
        "vhlo.return_v1"(%0) : (!vhlo.tensor_v1<!vhlo.f32_v1>) -> ()
      } {arg_attrs = #vhlo.array_v1<[]>, res_attrs = #vhlo.array_v1<[]>, sym_visibility = #vhlo.string_v1<"">}
    }
  )";
  WriteFile(JoinPath(dir, "serve.mlirbc"), mlir_content);

  auto result_or = LoadSlimModel(dir, &context_);
  ASSERT_TRUE(result_or.ok()) << result_or.status();
  mlir::OwningOpRef<mlir::ModuleOp> module = std::move(result_or.value());
  int num_dots = 0;
  module->walk([&](mlir::stablehlo::DotGeneralOp) { ++num_dots; });
  EXPECT_EQ(num_dots, 1);
}

TEST_F(SlimModelImporterTest, MissingMainFunction) {
  std::string dir = JoinPath(testing::TempDir(), "missing_main_dir");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  std::string metadata_path = JoinPath(dir, "weights_metadata.json");
  WriteFile(metadata_path, R"({"signatures": {"serve": []}})");

  std::string mlir_content = R"(
    module {
      func.func @other_func(%arg0: tensor<1xf32>) -> tensor<1xf32> {
        return %arg0 : tensor<1xf32>
      }
    }
  )";
  WriteFile(JoinPath(dir, "serve.mlirbc"), mlir_content);

  auto result = LoadSlimModel(dir, &context_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(SlimModelImporterTest, DuplicateArgIndexInMetadata) {
  std::string dir = JoinPath(testing::TempDir(), "dup_arg_index_dir");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  std::string metadata_json = R"({
    "signatures": {
      "serve": [
        {"arg_index": 1, "offset": 0},
        {"arg_index": 1, "offset": 0}
      ]
    }
  })";
  WriteFile(JoinPath(dir, "weights_metadata.json"), metadata_json);

  float weight_val = 3.14f;
  std::string params_data(reinterpret_cast<const char*>(&weight_val),
                          sizeof(float));
  WriteFile(JoinPath(dir, "params.bin"), params_data);

  std::string mlir_content = R"(
    module {
      func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
        return %0 : tensor<1xf32>
      }
    }
  )";
  WriteFile(JoinPath(dir, "serve.mlirbc"), mlir_content);

  auto result_or = LoadSlimModel(dir, &context_);
  ASSERT_TRUE(result_or.ok());
  auto module = std::move(result_or.value());
  auto serve_func = module->lookupSymbol<mlir::func::FuncOp>("serve");
  ASSERT_NE(serve_func, nullptr);
  EXPECT_EQ(serve_func.getNumArguments(), 1);
}

TEST_F(SlimModelImporterTest, DynamicOrUnrankedWeightArg) {
  std::string dir = JoinPath(testing::TempDir(), "dynamic_weight_arg_dir");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  std::string metadata_json = R"({
    "signatures": {
      "serve": [
        {"arg_index": 0, "offset": 0}
      ]
    }
  })";
  WriteFile(JoinPath(dir, "weights_metadata.json"), metadata_json);

  float weight_val = 3.14f;
  std::string params_data(reinterpret_cast<const char*>(&weight_val),
                          sizeof(float));
  WriteFile(JoinPath(dir, "params.bin"), params_data);

  std::string mlir_content = R"(
    module {
      func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
        return %arg0 : tensor<*xf32>
      }
    }
  )";
  WriteFile(JoinPath(dir, "serve.mlirbc"), mlir_content);

  auto result = LoadSlimModel(dir, &context_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(SlimModelImporterTest, InvalidArgIndexOutOfBounds) {
  std::string dir = JoinPath(testing::TempDir(), "out_of_bounds_arg_dir");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  std::string metadata_json = R"({
    "signatures": {
      "serve": [
        {"arg_index": 5, "offset": 0}
      ]
    }
  })";
  WriteFile(JoinPath(dir, "weights_metadata.json"), metadata_json);

  float weight_val = 3.14f;
  std::string params_data(reinterpret_cast<const char*>(&weight_val),
                          sizeof(float));
  WriteFile(JoinPath(dir, "params.bin"), params_data);

  std::string mlir_content = R"(
    module {
      func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
        return %arg0 : tensor<1xf32>
      }
    }
  )";
  WriteFile(JoinPath(dir, "serve.mlirbc"), mlir_content);

  auto result = LoadSlimModel(dir, &context_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(SlimModelImporterTest, WeightOffsetOutOfBounds) {
  std::string dir = JoinPath(testing::TempDir(), "weight_offset_oob_dir");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  std::string metadata_json = R"({
    "signatures": {
      "serve": [
        {"arg_index": 0, "offset": 1000}
      ]
    }
  })";
  WriteFile(JoinPath(dir, "weights_metadata.json"), metadata_json);

  float weight_val = 3.14f;
  std::string params_data(reinterpret_cast<const char*>(&weight_val),
                          sizeof(float));
  WriteFile(JoinPath(dir, "params.bin"), params_data);

  std::string mlir_content = R"(
    module {
      func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
        return %arg0 : tensor<1xf32>
      }
    }
  )";
  WriteFile(JoinPath(dir, "serve.mlirbc"), mlir_content);

  auto result = LoadSlimModel(dir, &context_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(SlimModelImporterTest, LargeWeightTensorUsesDenseResourceAndFadvise) {
  std::string dir = JoinPath(testing::TempDir(), "large_weight_slim_model");
  ASSERT_TRUE(tsl::Env::Default()->RecursivelyCreateDir(dir).ok());

  // Place a 4-byte small tensor at offset 0 and a 1 MiB (262144 x f32) tensor
  // at non-page-aligned offset 64 to exercise both DenseResourceElementsAttr
  // (num_elements > 64) and posix_fadvise (count >= 1 MiB).
  std::string metadata_json = R"({
    "signatures": {
      "serve": [
        {"arg_index": 1, "offset": 0},
        {"arg_index": 2, "offset": 64}
      ]
    },
    "signature_inputs": {
      "serve": ["x"]
    },
    "signature_outputs": {
      "serve": ["y"]
    }
  })";
  WriteFile(JoinPath(dir, "weights_metadata.json"), metadata_json);

  constexpr size_t kLargeElements = 262144;
  constexpr size_t kLargeOffset = 64;
  std::string params_data(kLargeOffset + kLargeElements * sizeof(float), '\0');
  float small_val = 2.5f;
  std::copy_n(reinterpret_cast<const char*>(&small_val), sizeof(float),
              params_data.data());
  std::vector<float> large_vals(kLargeElements);
  for (size_t i = 0; i < kLargeElements; ++i) {
    large_vals[i] = static_cast<float>(i % 1024) * 0.25f;
  }
  std::copy_n(reinterpret_cast<const char*>(large_vals.data()),
              kLargeElements * sizeof(float),
              params_data.data() + kLargeOffset);
  WriteFile(JoinPath(dir, "params.bin"), params_data);

  std::string mlir_content = R"(
    module {
      func.func @main(%arg0: tensor<262144xf32>, %arg1: tensor<1xf32>, %arg2: tensor<262144xf32>) -> tensor<262144xf32> {
        %0 = "stablehlo.add"(%arg0, %arg2) : (tensor<262144xf32>, tensor<262144xf32>) -> tensor<262144xf32>
        return %0 : tensor<262144xf32>
      }
    }
  )";
  WriteFile(JoinPath(dir, "serve.mlirbc"), mlir_content);

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> module_status =
      LoadSlimModel(dir, &context_);
  ASSERT_TRUE(module_status.ok()) << module_status.status();

  mlir::OwningOpRef<mlir::ModuleOp> module = *std::move(module_status);
  mlir::func::FuncOp serve_func =
      module->lookupSymbol<mlir::func::FuncOp>("serve");
  ASSERT_NE(serve_func, nullptr);
  EXPECT_EQ(serve_func.getNumArguments(), 1);

  int resource_const_count = 0;
  serve_func.walk([&](mlir::stablehlo::ConstantOp const_op) {
    if (auto resource_attr = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(
            const_op.getValue())) {
      ++resource_const_count;
      mlir::AsmResourceBlob* blob = resource_attr.getRawHandle().getBlob();
      ASSERT_NE(blob, nullptr);
      llvm::ArrayRef<char> data = blob->getData();
      ASSERT_EQ(data.size(), kLargeElements * sizeof(float));
      const float* float_data = reinterpret_cast<const float*>(data.data());
      EXPECT_FLOAT_EQ(float_data[0], large_vals[0]);
      EXPECT_FLOAT_EQ(float_data[12345], large_vals[12345]);
      EXPECT_FLOAT_EQ(float_data[kLargeElements - 1],
                      large_vals[kLargeElements - 1]);
    }
  });
  EXPECT_EQ(resource_const_count, 1);
}

}  // namespace
}  // namespace tensorflow
