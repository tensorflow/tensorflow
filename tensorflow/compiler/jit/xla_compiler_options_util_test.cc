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

#include "tensorflow/compiler/jit/xla_compiler_options_util.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/pjrt_device_compiler_client.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/client_library.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {
using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
using XlaDeviceExecutablePersistor =
    DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>;
using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;
using PjRtDeviceExecutablePersistor =
    DeviceExecutablePersistor<xla::PjRtLoadedExecutable, xla::PjRtClient>;

XlaDeviceCompiler* CreateXlaDeviceCompiler(DeviceType device_type,
                                           xla::LocalClient* local_client) {
  auto persistor = std::make_unique<XlaDeviceExecutablePersistor>(
      XlaDeviceExecutablePersistor::Config(), device_type);
  auto compiler_client =
      std::make_unique<XlaDeviceCompilerClient>(local_client);
  return new XlaDeviceCompiler(std::move(persistor),
                               std::move(compiler_client));
}

PjRtDeviceCompiler* CreatePjRtDeviceCompiler(DeviceType device_type,
                                             xla::PjRtClient* pjrt_client) {
  auto persistor = std::make_unique<PjRtDeviceExecutablePersistor>(
      PjRtDeviceExecutablePersistor::Config(), device_type);
  auto compiler_client =
      std::make_unique<PjRtDeviceCompilerClient>(pjrt_client);
  return new PjRtDeviceCompiler(std::move(persistor),
                                std::move(compiler_client));
}

std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
GetShapeDeterminationFns() {
  XlaHelpers::ShapeRepresentationFn shape_representation_fn =
      [](const TensorShape&, DataType, bool, XlaLayoutPreference) {
        return xla::Shape();
      };
  XlaShapeLayoutHelpers::LayoutPreferenceFn layout_preference_fn =
      [](const TensorShape&, DataType, std::optional<XlaArgument::Kind>) {
        return tensorflow::XlaLayoutPreference::kTpuPreferLinearLayout;
      };
  return {XlaShapeLayoutHelpers::ShapeDeterminationFns{
      layout_preference_fn, shape_representation_fn}};
}

std::unique_ptr<XlaDevice::Metadata> CreateXlaDeviceMetadata(
    DeviceType compilation_device_type) {
  return std::make_unique<XlaDevice::Metadata>(
      /*device_ordinal=*/0, /*platform=*/nullptr, compilation_device_type,
      GetShapeDeterminationFns(), XlaDevice::PaddedShapeFn(),
      /*use_multiple_streams=*/false);
}

std::unique_ptr<PjRtBaseDevice::Metadata> CreatePjRtDeviceMetadata(
    DeviceType compilation_device_type) {
  return std::make_unique<PjRtBaseDevice::Metadata>(compilation_device_type,
                                                    GetShapeDeterminationFns());
}

class XlaCompilerOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  }

  DeviceSetup device_setup_;
};

TEST_F(XlaCompilerOptionsTest, PjRtOptionsXlaDevice) {
  device_setup_.AddDevicesAndSetUp({DEVICE_XLA_GPU});
  Device* device = device_setup_.GetDevice(DEVICE_XLA_GPU);
  DeviceType compilation_device_type = DeviceType(DEVICE_GPU_XLA_JIT);

  se::Platform::Id platform_id = nullptr;
  auto xla_device_metadata = CreateXlaDeviceMetadata(compilation_device_type);
  std::shared_ptr<se::DeviceMemoryAllocator> custom_allocator;
  XlaPlatformInfo platform_info(
      compilation_device_type, platform_id, xla_device_metadata.get(),
      /*pjrt_device_metadata=*/nullptr, custom_allocator);

  XlaCompiler::Options options = GenerateCompilerOptionsForPjRt(
      *device_setup_.flr(), device, platform_info,
      /*pjrt_device_compiler=*/nullptr);

  EXPECT_EQ(options.device_type, compilation_device_type);
  EXPECT_EQ(options.device_ordinal, 0);
  EXPECT_NE(options.flib_def, nullptr);
  EXPECT_EQ(options.graph_def_version, TF_GRAPH_DEF_VERSION);
  EXPECT_FALSE(options.allow_cpu_custom_calls);
  EXPECT_FALSE(options.alias_passthrough_params);
  // Check if options have the supplied shape determination functions set.
  TF_ASSERT_OK_AND_ASSIGN(
      auto shape, options.shape_determination_fns.shape_representation_fn(
                      TensorShape(), DT_FLOAT, false,
                      tensorflow::XlaLayoutPreference::kTpuPreferLinearLayout));
  EXPECT_EQ(shape, xla::Shape());
  EXPECT_EQ(options.shape_determination_fns.layout_preference_fn(
                TensorShape(), DT_FLOAT, std::nullopt),
            tensorflow::XlaLayoutPreference::kTpuPreferLinearLayout);
}

TEST_F(XlaCompilerOptionsTest, PjRtOptionsPjRtBaseDevice) {
  // Although DEVICE_CPU isn't a PjRtBaseDevice, we use it here just for testing
  // purposes and to keep things simple. Creating a TpuDevice or
  // NextPluggableDevice in the context of this unit test is non-trivial.
  device_setup_.AddDevicesAndSetUp({DEVICE_CPU});
  Device* device = device_setup_.GetDevice(DEVICE_CPU);
  DeviceType compilation_device_type = DeviceType(DEVICE_CPU_XLA_JIT);

  auto pjrt_device_metadata = CreatePjRtDeviceMetadata(compilation_device_type);
  XlaPlatformInfo platform_info(
      compilation_device_type, /*platform_id=*/nullptr,
      /*xla_device_metadata=*/nullptr,
      /*pjrt_device_metadata=*/pjrt_device_metadata.get(),
      /*device_allocator=*/nullptr);

  XlaCompiler::Options options = GenerateCompilerOptionsForPjRt(
      *device_setup_.flr(), device, platform_info,
      /*pjrt_device_compiler=*/nullptr);

  EXPECT_EQ(options.device_type, compilation_device_type);
  EXPECT_EQ(options.device_ordinal, 0);
  EXPECT_NE(options.flib_def, nullptr);
  EXPECT_EQ(options.graph_def_version, TF_GRAPH_DEF_VERSION);
  EXPECT_FALSE(options.allow_cpu_custom_calls);
  EXPECT_FALSE(options.alias_passthrough_params);
  // Check if options have the supplied shape determination functions set.
  TF_ASSERT_OK_AND_ASSIGN(
      auto shape, options.shape_determination_fns.shape_representation_fn(
                      TensorShape(), DT_FLOAT, false,
                      tensorflow::XlaLayoutPreference::kTpuPreferLinearLayout));
  EXPECT_EQ(shape, xla::Shape());
  EXPECT_EQ(options.shape_determination_fns.layout_preference_fn(
                TensorShape(), DT_FLOAT, std::nullopt),
            tensorflow::XlaLayoutPreference::kTpuPreferLinearLayout);
}

TEST_F(XlaCompilerOptionsTest, PjRtOptionsNonXlaDevice) {
  device_setup_.AddDevicesAndSetUp({DEVICE_CPU});
  Device* device = device_setup_.GetDevice(DEVICE_CPU);
  DeviceType compilation_device_type = DeviceType(DEVICE_CPU_XLA_JIT);

  XlaPlatformInfo platform_info(compilation_device_type,
                                /*platform_id=*/nullptr,
                                /*xla_device_metadata=*/nullptr,
                                /*pjrt_device_metadata=*/nullptr,
                                /*device_allocator=*/nullptr);

  auto pjrt_device_compiler =
      CreatePjRtDeviceCompiler(compilation_device_type, nullptr);
  core::ScopedUnref pjrt_device_compiler_ref(pjrt_device_compiler);

  XlaCompiler::Options options = GenerateCompilerOptionsForPjRt(
      *device_setup_.flr(), device, platform_info, pjrt_device_compiler);

  EXPECT_EQ(options.device_type, compilation_device_type);
  EXPECT_EQ(options.device_ordinal, 0);
  EXPECT_NE(options.flib_def, nullptr);
  EXPECT_EQ(options.graph_def_version, TF_GRAPH_DEF_VERSION);
  EXPECT_FALSE(options.allow_cpu_custom_calls);
  EXPECT_FALSE(options.alias_passthrough_params);
  // Check whether options have default shape determination functions set.
  TF_ASSERT_OK_AND_ASSIGN(
      auto shape, options.shape_determination_fns.shape_representation_fn(
                      TensorShape(), DT_FLOAT, false,
                      tensorflow::XlaLayoutPreference::kNoPreference));
  xla::ShapeProto shape_proto;
  shape_proto.set_element_type(xla::PrimitiveType::F32);
  shape_proto.mutable_layout();
  EXPECT_EQ(shape, xla::Shape(shape_proto));
  EXPECT_EQ(options.shape_determination_fns.layout_preference_fn(
                TensorShape(), DT_FLOAT, std::nullopt),
            tensorflow::XlaLayoutPreference::kNoPreference);
}

TEST_F(XlaCompilerOptionsTest, XlaOptions) {
  device_setup_.AddDevicesAndSetUp({DEVICE_XLA_GPU});
  Device* device = device_setup_.GetDevice(DEVICE_XLA_GPU);

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  DeviceType device_type = DeviceType(DEVICE_XLA_GPU);
  DeviceType compilation_device_type = DeviceType(DEVICE_GPU_XLA_JIT);

  auto xla_device_compiler =
      CreateXlaDeviceCompiler(compilation_device_type, client);
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  se::Platform::Id platform_id = se::host::kHostPlatformId;
  auto xla_device_metadata = CreateXlaDeviceMetadata(compilation_device_type);
  std::shared_ptr<se::DeviceMemoryAllocator> custom_allocator;
  XlaPlatformInfo platform_info(
      device_type, platform_id, xla_device_metadata.get(),
      /*pjrt_device_metadata=*/nullptr, custom_allocator);

  XlaCompiler::Options options =
      GenerateCompilerOptions(*xla_device_compiler, *device_setup_.flr(),
                              device, nullptr, platform_info, false);

  EXPECT_EQ(options.device_type, compilation_device_type);
  EXPECT_NE(options.flib_def, nullptr);
  EXPECT_EQ(options.graph_def_version, TF_GRAPH_DEF_VERSION);
  EXPECT_TRUE(options.allow_cpu_custom_calls);
  EXPECT_NE(options.device_allocator, nullptr);
  EXPECT_FALSE(options.alias_passthrough_params);
  // Check if options have the supplied shape determination functions set.
  TF_ASSERT_OK_AND_ASSIGN(
      auto shape, options.shape_determination_fns.shape_representation_fn(
                      TensorShape(), DT_FLOAT, false,
                      tensorflow::XlaLayoutPreference::kTpuPreferLinearLayout));
  EXPECT_EQ(shape, xla::Shape());
  EXPECT_EQ(options.shape_determination_fns.layout_preference_fn(
                TensorShape(), DT_FLOAT, std::nullopt),
            tensorflow::XlaLayoutPreference::kTpuPreferLinearLayout);
}

TEST_F(XlaCompilerOptionsTest, XlaOptionsHasRefVarsNoXlaDeviceMetadata) {
  device_setup_.AddDevicesAndSetUp({DEVICE_CPU});
  Device* device = device_setup_.GetDevice(DEVICE_CPU);

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  DeviceType device_type = DeviceType(DEVICE_CPU);
  DeviceType compilation_device_type = DeviceType(DEVICE_CPU_XLA_JIT);

  auto xla_device_compiler =
      CreateXlaDeviceCompiler(compilation_device_type, client);
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  se::Platform::Id platform_id = se::host::kHostPlatformId;
  std::shared_ptr<se::DeviceMemoryAllocator> custom_allocator;
  XlaPlatformInfo platform_info(
      device_type, platform_id, /*xla_device_metadata=*/nullptr,
      /*pjrt_device_metadata=*/nullptr, custom_allocator);

  XlaCompiler::Options options =
      GenerateCompilerOptions(*xla_device_compiler, *device_setup_.flr(),
                              device, nullptr, platform_info, false);

  EXPECT_EQ(options.device_type, compilation_device_type);
  EXPECT_NE(options.flib_def, nullptr);
  EXPECT_EQ(options.graph_def_version, TF_GRAPH_DEF_VERSION);
  EXPECT_TRUE(options.allow_cpu_custom_calls);
  EXPECT_NE(options.device_allocator, nullptr);
  EXPECT_TRUE(options.alias_passthrough_params);
  // Check whether options have default shape determination functions set.
  TF_ASSERT_OK_AND_ASSIGN(
      auto shape, options.shape_determination_fns.shape_representation_fn(
                      TensorShape(), DT_FLOAT, false,
                      tensorflow::XlaLayoutPreference::kNoPreference));
  xla::ShapeProto shape_proto;
  shape_proto.set_element_type(xla::PrimitiveType::F32);
  shape_proto.mutable_layout();
  EXPECT_EQ(shape, xla::Shape(shape_proto));
  EXPECT_EQ(options.shape_determination_fns.layout_preference_fn(
                TensorShape(), DT_FLOAT, std::nullopt),
            tensorflow::XlaLayoutPreference::kNoPreference);
}

TEST_F(XlaCompilerOptionsTest, TfRtTpuOptions) {
  device_setup_.AddDevicesAndSetUp({DEVICE_TPU_NODE});

  // Just use the default local client for testing purposes.
  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  DeviceType compilation_device_type = DeviceType(DEVICE_TPU_XLA_JIT);

  auto xla_device_compiler =
      CreateXlaDeviceCompiler(compilation_device_type, client);
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  XlaCompiler::Options options = GenerateCompilerOptionsForTfrtTpu(
      *xla_device_compiler, *device_setup_.flr());

  EXPECT_EQ(options.device_type, compilation_device_type);
  EXPECT_NE(options.flib_def, nullptr);
  EXPECT_EQ(options.graph_def_version, TF_GRAPH_DEF_VERSION);
  EXPECT_FALSE(options.allow_cpu_custom_calls);
  EXPECT_FALSE(options.alias_passthrough_params);
}

TEST_F(XlaCompilerOptionsTest, GenerateCompileOptions) {
  XlaCompiler::CompileOptions option1 = GenerateCompileOptions(
      /*has_ref_vars=*/false, /*may_alias_resource_update=*/false);
  EXPECT_TRUE(option1.is_entry_computation);
  EXPECT_FALSE(option1.always_return_tuple);
  EXPECT_FALSE(option1.alias_resource_update);

  XlaCompiler::CompileOptions option2 = GenerateCompileOptions(
      /*has_ref_vars=*/false, /*may_alias_resource_update=*/true);
  EXPECT_TRUE(option2.alias_resource_update);

  XlaCompiler::CompileOptions option3 = GenerateCompileOptions(
      /*has_ref_vars=*/true, /*may_alias_resource_update=*/false);
  EXPECT_FALSE(option3.alias_resource_update);

  XlaCompiler::CompileOptions option4 = GenerateCompileOptions(
      /*has_ref_vars=*/true, /*may_alias_resource_update=*/true);
  EXPECT_FALSE(option4.alias_resource_update);
}

}  // namespace
}  // namespace tensorflow
