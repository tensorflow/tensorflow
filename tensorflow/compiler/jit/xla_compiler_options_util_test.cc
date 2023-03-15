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
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
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

XlaDeviceCompiler* CreateXlaDeviceCompiler(
    const XlaDeviceExecutablePersistor::Config& persistor_config,
    DeviceType device_type, xla::LocalClient* local_client) {
  auto persistor = std::make_unique<XlaDeviceExecutablePersistor>(
      std::move(persistor_config), device_type);
  auto compiler_client =
      std::make_unique<XlaDeviceCompilerClient>(local_client);
  return new XlaDeviceCompiler(std::move(persistor),
                               std::move(compiler_client));
}

std::unique_ptr<XlaDevice::Metadata> CreateXlaDeviceMetadata(
    DeviceType compilation_device_type) {
  XlaHelpers::ShapeRepresentationFn shape_representation_fn =
      [](const TensorShape&, DataType, bool, XlaLayoutPreference) {
        return xla::Shape();
      };
  XlaShapeLayoutHelpers::LayoutPreferenceFn layout_preference_fn =
      [](const TensorShape&, DataType, std::optional<XlaArgument::Kind>) {
        return tensorflow::XlaLayoutPreference::kTpuPreferLinearLayout;
      };
  std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
      shape_determination_fns = {XlaShapeLayoutHelpers::ShapeDeterminationFns{
          layout_preference_fn, shape_representation_fn}};
  return std::make_unique<XlaDevice::Metadata>(
      /*device_ordinal=*/0, /*platform=*/nullptr, compilation_device_type,
      shape_determination_fns, XlaDevice::PaddedShapeFn(),
      /*use_multiple_streams=*/false);
}

class XlaCompilerOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  }

  void AddDevicesAndSetUp(const std::vector<std::string>& device_names) {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    for (const auto& device_name : device_names) {
      device_count->insert({device_name, 1});
    }

    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));
    device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));

    OptimizerOptions opts;
    lib_def_ = std::make_unique<FunctionLibraryDefinition>(
        OpRegistry::Global(), FunctionDefLibrary());
    pflr_ = std::make_unique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, lib_def_.get(), opts,
        /*default_thread_pool=*/nullptr, /*cluster_flr=*/nullptr);
    flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  }

  Device* GetXlaGpuDevice() {
    if (device_mgr_ == nullptr) {
      return nullptr;
    }

    Device* device;
    TF_CHECK_OK(device_mgr_->LookupDevice(
        "/job:localhost/replica:0/task:0/device:XLA_GPU:0", &device));
    return device;
  }

  FunctionLibraryRuntime* flr_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
};

TEST_F(XlaCompilerOptionsTest, PjRtOptionsXlaDevice) {
  AddDevicesAndSetUp({DEVICE_XLA_GPU});
  Device* device = GetXlaGpuDevice();
  DeviceType compilation_device_type = DeviceType(DEVICE_GPU_XLA_JIT);

  se::Platform::Id platform_id = nullptr;
  auto xla_device_metadata = CreateXlaDeviceMetadata(compilation_device_type);
  std::shared_ptr<se::DeviceMemoryAllocator> custom_allocator;
  XlaPlatformInfo platform_info(compilation_device_type, platform_id,
                                xla_device_metadata.get(), custom_allocator);

  XlaCompiler::Options options =
      GenerateCompilerOptionsForPjRt(*flr_, device, platform_info);

  EXPECT_EQ(options.device_type, compilation_device_type);
  EXPECT_EQ(options.device_ordinal, 0);
  EXPECT_NE(options.flib_def, nullptr);
  EXPECT_EQ(options.graph_def_version, TF_GRAPH_DEF_VERSION);
  EXPECT_FALSE(options.allow_cpu_custom_calls);
  EXPECT_FALSE(options.alias_passthrough_params);
  EXPECT_FALSE(options.detailed_logging);
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

TEST_F(XlaCompilerOptionsTest, XlaOptions) {
  AddDevicesAndSetUp({DEVICE_XLA_CPU});
  Device* device = device_mgr_->HostCPU();

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  DeviceType device_type = DeviceType(DEVICE_XLA_CPU);
  DeviceType compilation_device_type = DeviceType(DEVICE_CPU_XLA_JIT);

  auto xla_device_compiler = CreateXlaDeviceCompiler(
      XlaDeviceExecutablePersistor::Config(), compilation_device_type, client);
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  se::Platform::Id platform_id = se::host::kHostPlatformId;
  auto xla_device_metadata = CreateXlaDeviceMetadata(compilation_device_type);
  std::shared_ptr<se::DeviceMemoryAllocator> custom_allocator;
  XlaPlatformInfo platform_info(device_type, platform_id,
                                xla_device_metadata.get(), custom_allocator);

  XlaCompiler::Options options = GenerateCompilerOptions(
      *xla_device_compiler, *flr_, device, nullptr, platform_info, false);

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
  AddDevicesAndSetUp({DEVICE_CPU});
  Device* device = device_mgr_->HostCPU();

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  DeviceType device_type = DeviceType(DEVICE_CPU);
  DeviceType compilation_device_type = DeviceType(DEVICE_CPU_XLA_JIT);

  auto xla_device_compiler = CreateXlaDeviceCompiler(
      XlaDeviceExecutablePersistor::Config(), compilation_device_type, client);
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  se::Platform::Id platform_id = se::host::kHostPlatformId;
  std::shared_ptr<se::DeviceMemoryAllocator> custom_allocator;
  XlaPlatformInfo platform_info(device_type, platform_id, nullptr,
                                custom_allocator);

  XlaCompiler::Options options = GenerateCompilerOptions(
      *xla_device_compiler, *flr_, device, nullptr, platform_info, false);

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
  AddDevicesAndSetUp({DEVICE_TPU_NODE});

  // Just use the default local client for testing purposes.
  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  DeviceType compilation_device_type = DeviceType(DEVICE_TPU_XLA_JIT);

  auto xla_device_compiler = CreateXlaDeviceCompiler(
      XlaDeviceExecutablePersistor::Config(), compilation_device_type, client);
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  XlaCompiler::Options options =
      GenerateCompilerOptionsForTfrtTpu(*xla_device_compiler, *flr_);

  EXPECT_EQ(options.device_type, compilation_device_type);
  EXPECT_NE(options.flib_def, nullptr);
  EXPECT_EQ(options.graph_def_version, TF_GRAPH_DEF_VERSION);
  EXPECT_FALSE(options.allow_cpu_custom_calls);
  EXPECT_FALSE(options.alias_passthrough_params);
}

}  // namespace
}  // namespace tensorflow
