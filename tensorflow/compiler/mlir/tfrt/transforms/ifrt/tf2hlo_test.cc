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
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Ne;
using tsl::testing::StatusIs;

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

class Tf2HloTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::RegisterAllTensorFlowDialects(registry);
    context_ = std::make_unique<mlir::MLIRContext>(registry);
  }
  TfToHloCompiler tf_to_hlo_compiler_;
  std::unique_ptr<mlir::MLIRContext> context_;
};

TEST_F(Tf2HloTest, Empty) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_empty.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, context_.get());

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(mlir_module.get(), *client));
  TF_ASSERT_OK(UpdateCompileMetadata(compile_metadata, {}));

  xla::CpuTopologyDescription cpu_topology =
      xla::CpuTopologyDescription::Create(
          xla::CpuId(), xla::CpuName(), /*platform_version=*/"",
          /*devices=*/std::vector<std::unique_ptr<xla::PjRtDevice>>{},
          /*machine_attributes=*/std::vector<std::string>{});
  std::shared_ptr<xla::CpuTopologyDescription> cpu_topology_ptr =
      std::make_shared<xla::CpuTopologyDescription>(cpu_topology);

  std::vector<int> variable_arg_indices;
  Tf2HloArg arg{
      .module = mlir_module.get(),
      .input_dtypes_and_shapes = {},
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(cpu_topology_ptr),
      .platform_name = xla::CpuName(),
  };
  auto result = tf_to_hlo_compiler_.CompileTfToHlo(arg);

  TF_ASSERT_OK(result.status());
}

// Multiple input and multiple out.
TEST_F(Tf2HloTest, Tuple) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_tuple.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, context_.get());

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {1, 3}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {3, 1}});

  TF_ASSERT_OK_AND_ASSIGN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(mlir_module.get(), *client));
  TF_ASSERT_OK(UpdateCompileMetadata(compile_metadata, dtype_and_shapes));

  xla::CpuTopologyDescription cpu_topology =
      xla::CpuTopologyDescription::Create(
          xla::CpuId(), xla::CpuName(), /*platform_version=*/"",
          /*devices=*/std::vector<std::unique_ptr<xla::PjRtDevice>>{},
          /*machine_attributes=*/std::vector<std::string>{});
  std::shared_ptr<xla::CpuTopologyDescription> cpu_topology_ptr =
      std::make_shared<xla::CpuTopologyDescription>(cpu_topology);

  std::vector<int> variable_arg_indices;

  Tf2HloArg arg{
      .module = mlir_module.get(),
      .input_dtypes_and_shapes = dtype_and_shapes,
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(cpu_topology_ptr),
      .platform_name = xla::CpuName(),
  };

  auto result = tf_to_hlo_compiler_.CompileTfToHlo(arg);

  TF_ASSERT_OK(result.status());
}

// Spmd and device assignment is given
TEST_F(Tf2HloTest, Spmd) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_spmd_with_device_assignment.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, context_.get());

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {4, 64}});

  TF_ASSERT_OK_AND_ASSIGN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(mlir_module.get(), *client));
  TF_ASSERT_OK(UpdateCompileMetadata(compile_metadata, dtype_and_shapes));

  xla::CpuTopologyDescription cpu_topology =
      xla::CpuTopologyDescription::Create(
          xla::CpuId(), xla::CpuName(), /*platform_version=*/"",
          /*devices=*/std::vector<std::unique_ptr<xla::PjRtDevice>>{},
          /*machine_attributes=*/std::vector<std::string>{});
  std::shared_ptr<xla::CpuTopologyDescription> cpu_topology_ptr =
      std::make_shared<xla::CpuTopologyDescription>(cpu_topology);

  std::vector<int> variable_arg_indices;
  Tf2HloArg arg{
      .module = mlir_module.get(),
      .input_dtypes_and_shapes = dtype_and_shapes,
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(cpu_topology_ptr),
      .platform_name = xla::CpuName(),
  };

  auto result = tf_to_hlo_compiler_.CompileTfToHlo(arg);

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
TEST_F(Tf2HloTest, UsingDefaultDeviceAssignment) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_spmd_no_device_assignment.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, context_.get());

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {4, 64}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {64, 10}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {1, 4}});

  TF_ASSERT_OK_AND_ASSIGN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(mlir_module.get(), *client));
  TF_ASSERT_OK(UpdateCompileMetadata(compile_metadata, dtype_and_shapes));

  xla::CpuTopologyDescription cpu_topology =
      xla::CpuTopologyDescription::Create(
          xla::CpuId(), xla::CpuName(), /*platform_version=*/"",
          /*devices=*/std::vector<std::unique_ptr<xla::PjRtDevice>>{},
          /*machine_attributes=*/std::vector<std::string>{});
  std::shared_ptr<xla::CpuTopologyDescription> cpu_topology_ptr =
      std::make_shared<xla::CpuTopologyDescription>(cpu_topology);

  std::vector<int> variable_arg_indices;
  Tf2HloArg arg{
      .module = mlir_module.get(),
      .input_dtypes_and_shapes = dtype_and_shapes,
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(cpu_topology_ptr),
      .platform_name = xla::CpuName(),
  };

  auto result = tf_to_hlo_compiler_.CompileTfToHlo(arg);

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
TEST_F(Tf2HloTest, XlaCallHostCallback) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/xla_call_host_callback.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path,
                                            mlir::ParserConfig(context_.get()));

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_INT32, {1}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_INT32, {1}});

  TF_ASSERT_OK_AND_ASSIGN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(mlir_module.get(), *client));
  TF_ASSERT_OK(UpdateCompileMetadata(compile_metadata, dtype_and_shapes));

  xla::CpuTopologyDescription cpu_topology =
      xla::CpuTopologyDescription::Create(
          xla::CpuId(), xla::CpuName(), /*platform_version=*/"",
          /*devices=*/std::vector<std::unique_ptr<xla::PjRtDevice>>{},
          /*machine_attributes=*/std::vector<std::string>{});
  std::shared_ptr<xla::CpuTopologyDescription> cpu_topology_ptr =
      std::make_shared<xla::CpuTopologyDescription>(cpu_topology);

  std::vector<int> variable_arg_indices;
  Tf2HloArg arg{
      .module = mlir_module.get(),
      .input_dtypes_and_shapes = dtype_and_shapes,
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(cpu_topology_ptr),
      .platform_name = xla::CpuName(),
  };

  auto result = tf_to_hlo_compiler_.CompileTfToHlo(arg);

  TF_ASSERT_OK(result.status());

  ASSERT_EQ((*result).host_compute_metadata.device_to_host().size(), 1);
  ASSERT_EQ(
      (*result).host_compute_metadata.device_to_host().begin()->metadata_size(),
      2);
  ASSERT_EQ((*result).host_compute_metadata.host_to_device().size(), 0);
}

// On GPU enabled build, the compilation should pass. On a GPU disabled build,
// the compilation should fail with a correct error message.
TEST_F(Tf2HloTest, GpuCompile) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_gpu.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, context_.get());

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  xla::ifrt::MockClient mock_client;
  ON_CALL(mock_client, GetDefaultDeviceAssignment)
      .WillByDefault([]() -> absl::StatusOr<xla::DeviceAssignment> {
        return xla::DeviceAssignment(1, 1);
      });

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {}});

  TF_ASSERT_OK_AND_ASSIGN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(mlir_module.get(), mock_client));
  TF_ASSERT_OK(UpdateCompileMetadata(compile_metadata, dtype_and_shapes));

  std::vector<int> variable_arg_indices;
  Tf2HloArg arg{
      .module = mlir_module.get(),
      .input_dtypes_and_shapes = dtype_and_shapes,
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(
          std::make_shared<xla::StreamExecutorGpuTopologyDescription>(
              xla::CudaId(), xla::CudaName(), /*gpu_topology=*/nullptr)),
      .platform_name = xla::CudaName(),
  };

  auto result = tf_to_hlo_compiler_.CompileTfToHlo(arg);
#if defined(GOOGLE_CUDA)
  LOG(INFO) << "GPU compile success";
  EXPECT_OK(result);
#else
  LOG(INFO) << "Non-GPU compile failure";
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kUnimplemented,
                               HasSubstr("CUDA or ROCM build required")));
#endif
}

TEST_F(Tf2HloTest, SameArgProduceSameKeyFingerprint) {
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/xla_call_host_callback.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path,
                                            mlir::ParserConfig(context_.get()));

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_INT32, {1}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_INT32, {1}});

  TF_ASSERT_OK_AND_ASSIGN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(mlir_module.get(), *client));
  TF_ASSERT_OK(UpdateCompileMetadata(compile_metadata, dtype_and_shapes));

  xla::CpuTopologyDescription cpu_topology =
      xla::CpuTopologyDescription::Create(
          xla::CpuId(), xla::CpuName(), /*platform_version=*/"",
          /*devices=*/std::vector<std::unique_ptr<xla::PjRtDevice>>{},
          /*machine_attributes=*/std::vector<std::string>{});
  std::shared_ptr<xla::CpuTopologyDescription> cpu_topology_ptr =
      std::make_shared<xla::CpuTopologyDescription>(cpu_topology);

  std::vector<int> variable_arg_indices;
  Tf2HloArg arg0{
      .module = mlir_module.get(),
      .input_dtypes_and_shapes = dtype_and_shapes,
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(cpu_topology_ptr),
  };
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_clone =
      mlir::OwningOpRef<mlir::ModuleOp>(mlir_module->clone());
  Tf2HloArg arg1{
      .module = mlir_module_clone.get(),
      .input_dtypes_and_shapes = dtype_and_shapes,
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(cpu_topology_ptr),
  };

  TfToHloCompiler tf_to_hlo_compiler;
  TF_ASSERT_OK_AND_ASSIGN(std::string key0, tf_to_hlo_compiler.Key(arg0));
  TF_ASSERT_OK_AND_ASSIGN(std::string key1, tf_to_hlo_compiler.Key(arg1));

  EXPECT_THAT(key0, Eq(key1));
}

TEST_F(Tf2HloTest, DifferentCompileMetadataProduceDifferentKeyFingerprint) {
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/xla_call_host_callback.mlir"));

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path,
                                            mlir::ParserConfig(context_.get()));

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_INT32, {1}});
  dtype_and_shapes.push_back(DtypeAndShape{DT_INT32, {1}});

  TF_ASSERT_OK_AND_ASSIGN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(mlir_module.get(), *client));
  TF_ASSERT_OK(UpdateCompileMetadata(compile_metadata, dtype_and_shapes));

  xla::CpuTopologyDescription cpu_topology =
      xla::CpuTopologyDescription::Create(
          xla::CpuId(), xla::CpuName(), /*platform_version=*/"",
          /*devices=*/std::vector<std::unique_ptr<xla::PjRtDevice>>{},
          /*machine_attributes=*/std::vector<std::string>{});
  std::shared_ptr<xla::CpuTopologyDescription> cpu_topology_ptr =
      std::make_shared<xla::CpuTopologyDescription>(cpu_topology);

  std::vector<int> variable_arg_indices;
  Tf2HloArg arg0{
      .module = mlir_module.get(),
      .input_dtypes_and_shapes = dtype_and_shapes,
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(cpu_topology_ptr),
  };
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_clone =
      mlir::OwningOpRef<mlir::ModuleOp>(mlir_module->clone());
  compile_metadata.set_num_replicas(11111);
  Tf2HloArg arg1{
      .module = mlir_module_clone.get(),
      .input_dtypes_and_shapes = dtype_and_shapes,
      .variable_arg_indices = variable_arg_indices,
      .entry_function_name = "main",
      .compile_metadata = compile_metadata,
      .shape_representation_fn = tensorflow::IdentityShapeRepresentationFn(),
      .topology = std::make_shared<xla::ifrt::PjRtTopology>(cpu_topology_ptr),
  };

  TfToHloCompiler tf_to_hlo_compiler;

  TF_ASSERT_OK_AND_ASSIGN(std::string key0, tf_to_hlo_compiler.Key(arg0));
  TF_ASSERT_OK_AND_ASSIGN(std::string key1, tf_to_hlo_compiler.Key(arg1));
  EXPECT_THAT(key0, Ne(key1));
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
