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

#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

// TODO(b/229726259): Make EqualsProto available in OSS
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tsl::protobuf::Message& expected)
      : expected_(expected.SerializeAsString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener*) const {
    return p.SerializeAsString() == expected_;
  }

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tsl::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

TEST(Tf2HloTest, Empty) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_empty.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  auto result = CompileTfToHlo(mlir_module.get(), {}, "main", *client,
                               tensorflow::IdentityShapeRepresentationFn());

  TF_ASSERT_OK(result.status());
}

// Multiple input and multiple out.
TEST(Tf2HloTest, Tuple) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_tuple.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {1, 3}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {3, 1}});
  auto result =
      CompileTfToHlo(mlir_module.get(), dtype_and_shapes, "main", *client,
                     tensorflow::IdentityShapeRepresentationFn());

  TF_ASSERT_OK(result.status());
}

// Spmd and device assignment is given
TEST(Tf2HloTest, Spmd) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_spmd_with_device_assignment.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {4, 64}});
  auto result =
      CompileTfToHlo(mlir_module.get(), dtype_and_shapes, "main", *client,
                     tensorflow::IdentityShapeRepresentationFn());

  LOG(INFO) << result->compile_metadata;
  TF_ASSERT_OK(result.status());

  tensorflow::tpu::TPUCompileMetadataProto expected_compile_metadata;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        args {
          dtype: DT_FLOAT
          shape {
            dim { size: 4 }
            dim { size: 64 }
          }
          kind: PARAMETER
          sharding {
            type: OTHER
            tile_assignment_dimensions: 2
            tile_assignment_dimensions: 1
            tile_assignment_devices: 0
            tile_assignment_devices: 1
          }
          is_bounded_dynamic_dim: false
        }
        retvals { sharding {} }
        num_replicas: 1
        num_cores_per_replica: 2
        device_assignment {
          replica_count: 1
          computation_count: 2
          computation_devices { replica_device_ids: 0 }
          computation_devices { replica_device_ids: 1 }
        }
        use_spmd_for_xla_partitioning: true
        compile_options {}
      )pb",
      &expected_compile_metadata));

  EXPECT_THAT(result->compile_metadata, EqualsProto(expected_compile_metadata));
}

// Spmd and use default device assignment b/c no device assignment is given
TEST(Tf2HloTest, UsingDefaultDeviceAssignment) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_spmd_no_device_assignment.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {4, 64}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {64, 10}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {1, 4}});
  auto result =
      CompileTfToHlo(mlir_module.get(), dtype_and_shapes, "main", *client,
                     tensorflow::IdentityShapeRepresentationFn());

  LOG(INFO) << result->compile_metadata;
  TF_ASSERT_OK(result.status());

  tensorflow::tpu::TPUCompileMetadataProto expected_compile_metadata;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        args {
          dtype: DT_FLOAT
          shape {
            dim { size: 4 }
            dim { size: 64 }
          }
          kind: PARAMETER
          sharding {
            type: OTHER
            tile_assignment_dimensions: 2
            tile_assignment_dimensions: 1
            tile_assignment_devices: 0
            tile_assignment_devices: 1
          }
          is_bounded_dynamic_dim: false
        }
        args {
          dtype: DT_FLOAT
          shape {
            dim { size: 64 }
            dim { size: 10 }
          }
          kind: PARAMETER
          sharding {
            type: OTHER
            tile_assignment_dimensions: 2
            tile_assignment_dimensions: 1
            tile_assignment_devices: 0
            tile_assignment_devices: 1
          }
          is_bounded_dynamic_dim: false
        }
        args {
          dtype: DT_FLOAT
          shape {
            dim { size: 1 }
            dim { size: 4 }
          }
          kind: PARAMETER
          is_bounded_dynamic_dim: false
        }
        retvals { sharding {} }
        num_replicas: 1
        num_cores_per_replica: 2
        device_assignment {
          replica_count: 1
          computation_count: 2
          computation_devices { replica_device_ids: 0 }
          computation_devices { replica_device_ids: 1 }
        }
        use_spmd_for_xla_partitioning: true
        compile_options {}
      )pb",
      &expected_compile_metadata));

  EXPECT_THAT(result->compile_metadata, EqualsProto(expected_compile_metadata));
}

// Multiple input and multiple out.
TEST(Tf2HloTest, XlaCallHostCallback) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/xla_call_host_callback.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path,
                                            mlir::ParserConfig(&context));

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_INT32, {1}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_INT32, {1}});

  auto result =
      CompileTfToHlo(mlir_module.get(), dtype_and_shapes, "main", *client,
                     tensorflow::IdentityShapeRepresentationFn());

  TF_ASSERT_OK(result.status());

  ASSERT_EQ((*result).host_compute_metadata.device_to_host().size(), 1);
  ASSERT_EQ(
      (*result).host_compute_metadata.device_to_host().begin()->metadata_size(),
      2);
  ASSERT_EQ((*result).host_compute_metadata.host_to_device().size(), 0);
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
