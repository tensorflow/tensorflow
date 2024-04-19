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

#include "tensorflow/compiler/jit/xla_platform_info.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {
using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;

class XlaPlatformInfoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
    tensorflow::GetMarkForCompilationPassFlags()
        ->tf_xla_persistent_cache_directory = "";
    tensorflow::GetMarkForCompilationPassFlags()
        ->tf_xla_persistent_cache_device_types = "";
  }

  DeviceSetup device_setup_;
};

class StubDevice : public DeviceBase {
 public:
  StubDevice() : DeviceBase(nullptr) {}
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TEST_F(XlaPlatformInfoTest, BuildXlaDeviceCompilerXlaDeviceMetadata) {
  device_setup_.AddDevicesAndSetUp({DEVICE_XLA_GPU});

  Device* device = device_setup_.GetDevice(DEVICE_XLA_GPU);
  const XlaDevice::Metadata* metadata = nullptr;
  TF_CHECK_OK(XlaDevice::GetMetadataFromDevice(device, &metadata));
  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(device);

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceType compilation_device_type,
      GetCompilationDeviceType(platform_info.device_type()));

  XlaDeviceCompiler* xla_device_compiler = nullptr;
  TF_EXPECT_OK(BuildXlaDeviceCompiler(device, device_setup_.flr(),
                                      platform_info, compilation_device_type,
                                      &xla_device_compiler));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  EXPECT_EQ(xla_device_compiler->device_type(), metadata->jit_device_type());
  EXPECT_EQ(xla_device_compiler->client(), metadata->client());
}

TEST_F(XlaPlatformInfoTest, BuildXlaDeviceCompilerXlaDeviceCacheEnabled) {
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_directory = "/tmp/xla_cache";
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_device_types = DEVICE_XLA_GPU;
  device_setup_.AddDevicesAndSetUp({DEVICE_XLA_GPU});

  Device* device = device_setup_.GetDevice(DEVICE_XLA_GPU);
  const XlaDevice::Metadata* metadata = nullptr;
  TF_CHECK_OK(XlaDevice::GetMetadataFromDevice(device, &metadata));
  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(device);

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceType compilation_device_type,
      GetCompilationDeviceType(platform_info.device_type()));

  XlaDeviceCompiler* xla_device_compiler = nullptr;
  TF_EXPECT_OK(BuildXlaDeviceCompiler(device, device_setup_.flr(),
                                      platform_info, compilation_device_type,
                                      &xla_device_compiler));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  EXPECT_EQ(xla_device_compiler->device_type(), metadata->jit_device_type());
  EXPECT_EQ(xla_device_compiler->client(), metadata->client());
  EXPECT_EQ(xla_device_compiler->persistor()->persistent_cache_directory(),
            "/tmp/xla_cache");
}

TEST_F(XlaPlatformInfoTest, BuildXlaDeviceCompilerNonXlaDevice) {
  device_setup_.AddDevicesAndSetUp({DEVICE_GPU});
  Device* device = device_setup_.GetDevice(DEVICE_GPU);

  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(device);
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceType compilation_device_type,
      GetCompilationDeviceType(platform_info.device_type()));
  XlaDeviceCompiler* xla_device_compiler = nullptr;
  TF_EXPECT_OK(BuildXlaDeviceCompiler(device, device_setup_.flr(),
                                      platform_info, compilation_device_type,
                                      &xla_device_compiler));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  EXPECT_EQ(xla_device_compiler->device_type(), DeviceType(DEVICE_GPU_XLA_JIT));
  EXPECT_TRUE(xla_device_compiler->client() != nullptr);
}

TEST_F(XlaPlatformInfoTest, GetOrCreatePjRtDeviceCompilerAndProfilerXlaDevice) {
  DeviceType device_type = DeviceType(DEVICE_XLA_GPU);
  device_setup_.AddDevicesAndSetUp({device_type.type()});

  Device* device = device_setup_.GetDevice(device_type.type());
  const XlaDevice::Metadata* metadata = nullptr;
  TF_CHECK_OK(XlaDevice::GetMetadataFromDevice(device, &metadata));
  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(device);

  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  params.device = device;
  OpKernelContext ctx(&params, 0);

  PjRtDeviceCompiler* pjrt_device_compiler = nullptr;
  DeviceCompilationProfiler* profiler = nullptr;
  TF_EXPECT_OK(GetOrCreatePjRtDeviceCompilerAndProfiler(
      ctx, platform_info, device_setup_.flr(), &pjrt_device_compiler,
      &profiler));
  core::ScopedUnref pjrt_device_compiler_ref(pjrt_device_compiler);
  core::ScopedUnref profiler_ref(profiler);

  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, GetOrCreatePjRtClient(device_type));
  EXPECT_EQ(pjrt_device_compiler->device_type(), metadata->jit_device_type());
  EXPECT_EQ(pjrt_device_compiler->client(), pjrt_client);
}

TEST_F(XlaPlatformInfoTest,
       GetOrCreatePjRtDeviceCompilerAndProfilerGpuDeviceCacheEnabled) {
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_directory = "/tmp/xla_cache";
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_device_types = DEVICE_GPU_XLA_JIT;
  device_setup_.AddDevicesAndSetUp({DEVICE_GPU});
  Device* device = device_setup_.GetDevice(DEVICE_GPU);
  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(device);

  ResourceMgr resource_mgr("");
  OpKernelContext::Params params;
  params.resource_manager = &resource_mgr;
  params.device = device;
  OpKernelContext ctx(&params, 0);

  PjRtDeviceCompiler* pjrt_device_compiler = nullptr;
  DeviceCompilationProfiler* profiler = nullptr;
  TF_EXPECT_OK(GetOrCreatePjRtDeviceCompilerAndProfiler(
      ctx, platform_info, device_setup_.flr(), &pjrt_device_compiler,
      &profiler));
  EXPECT_EQ(pjrt_device_compiler->persistor()->persistent_cache_directory(),
            "/tmp/xla_cache");
  core::ScopedUnref pjrt_device_compiler_ref(pjrt_device_compiler);
  core::ScopedUnref profiler_ref(profiler);
}
#endif

TEST_F(XlaPlatformInfoTest, BuildXlaDeviceCompilerTpuDevice) {
  DeviceType compilation_device_type = DeviceType(DEVICE_TPU_XLA_JIT);

  // Instead of creating/initializing a TPU device, create a dummy platform_info
  // and use a nullptr for Device for testing purposes. Only
  // XlaPlatformInfo::device_type() is needed to build the appropriate
  // XlaDeviceCompiler.
  Device* device = nullptr;
  XlaPlatformInfo platform_info(DeviceType(DEVICE_TPU), /*platform_id=*/nullptr,
                                /*xla_device_metadata=*/nullptr,
                                /*pjrt_device_metadata=*/nullptr,
                                /*device_allocator=*/nullptr);

  XlaDeviceCompiler* xla_device_compiler = nullptr;
  TF_EXPECT_OK(BuildXlaDeviceCompiler(device, nullptr, platform_info,
                                      compilation_device_type,
                                      &xla_device_compiler));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  EXPECT_EQ(xla_device_compiler->device_type(), compilation_device_type);
  // TFRT-TPU is used if device type is `DEVICE_TPU` and `platform_info` does
  // not have `xla_device_metadata`. XlaDeviceCompiler/xla::LocalClient is not
  // used in this case.
  EXPECT_EQ(xla_device_compiler->client(), nullptr);
}

TEST_F(XlaPlatformInfoTest, BuildXlaDeviceCompilerNoCompilationCache) {
  DeviceType compilation_device_type = DeviceType(DEVICE_TPU_XLA_JIT);
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_directory = "/tmp/xla_cache";
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_device_types = DEVICE_XLA_GPU;

  // Instead of creating/initializing a TPU device, create a dummy platform_info
  // and use a nullptr for Device for testing purposes. Only
  // XlaPlatformInfo::device_type() is needed to build the appropriate
  // XlaDeviceCompiler.
  Device* device = nullptr;
  XlaPlatformInfo platform_info(DeviceType(DEVICE_TPU), /*platform_id=*/nullptr,
                                /*xla_device_metadata=*/nullptr,
                                /*pjrt_device_metadata=*/nullptr,
                                /*device_allocator=*/nullptr);

  XlaDeviceCompiler* xla_device_compiler = nullptr;
  TF_EXPECT_OK(BuildXlaDeviceCompiler(device, nullptr, platform_info,
                                      compilation_device_type,
                                      &xla_device_compiler));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  EXPECT_EQ(xla_device_compiler->device_type(), compilation_device_type);
  // Check to make sure compilation cache path is empty.
  EXPECT_TRUE(
      xla_device_compiler->persistor()->persistent_cache_directory().empty());
}

// TODO(b/255826209): Look into using an actual TPU device for the unit test,
// and move this out of OSS.
TEST_F(XlaPlatformInfoTest,
       GetOrCreatePjRtDeviceCompilerAndProfilerTpuDeviceNoCompilationCache) {
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_directory = "/tmp/xla_cache";
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_device_types = DEVICE_GPU_XLA_JIT;
  DeviceType device_type = DeviceType(DEVICE_TPU);
  DeviceType compilation_device_type = DeviceType(DEVICE_TPU_XLA_JIT);
  // Use a CPU PjRtClient instead of a TPU one just for testing whether
  // GetOrCreatePjRtClient() is being called with the correct arguments.
  TF_CHECK_OK(SetPjRtClientInTFGlobalResourceManager(
      device_type,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1)
          .value()));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, GetOrCreatePjRtClient(device_type));

  // Instead of creating/initializing a TPU device, create a dummy platform_info
  // for testing purposes. Only XlaPlatformInfo::device_type() is needed to
  // build the appropriate PjRtDeviceCompiler.
  XlaPlatformInfo platform_info(device_type, /*platform_id=*/nullptr,
                                /*xla_device_metadata=*/nullptr,
                                /*pjrt_device_metadata=*/nullptr,
                                /*device_allocator=*/nullptr);

  OpKernelContext::Params params;
  StubDevice stub_device;
  params.device = &stub_device;
  OpKernelContext ctx(&params, 0);

  PjRtDeviceCompiler* pjrt_device_compiler = nullptr;
  DeviceCompilationProfiler* profiler = nullptr;
  TF_EXPECT_OK(GetOrCreatePjRtDeviceCompilerAndProfiler(
      ctx, platform_info, nullptr, &pjrt_device_compiler, &profiler));
  core::ScopedUnref pjrt_device_compiler_ref(pjrt_device_compiler);
  core::ScopedUnref profiler_ref(profiler);

  EXPECT_EQ(pjrt_device_compiler->device_type(), compilation_device_type);
  EXPECT_EQ(pjrt_device_compiler->client(), pjrt_client);
  EXPECT_TRUE(
      pjrt_device_compiler->persistor()->persistent_cache_directory().empty());
}

TEST_F(XlaPlatformInfoTest, GetPersistentCacheDirectoryMultiple) {
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_directory = "/tmp/xla_cache";
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_device_types = "GPU,CPU";
  DeviceType device_gpu = DeviceType(DEVICE_GPU);
  EXPECT_EQ(GetPersistentCacheDirectory(device_gpu), "/tmp/xla_cache");
  DeviceType device_cpu = DeviceType(DEVICE_CPU);
  EXPECT_EQ(GetPersistentCacheDirectory(device_cpu), "/tmp/xla_cache");
  DeviceType device_tpu = DeviceType(DEVICE_TPU);
  EXPECT_TRUE(GetPersistentCacheDirectory(device_tpu).empty());
}

TEST_F(XlaPlatformInfoTest, GetPersistentCacheDirectoryNoDeviceTypes) {
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_directory = "/tmp/xla_cache";
  tensorflow::GetMarkForCompilationPassFlags()
      ->tf_xla_persistent_cache_device_types = "";
  DeviceType device_gpu = DeviceType(DEVICE_GPU);
  EXPECT_EQ(GetPersistentCacheDirectory(device_gpu), "/tmp/xla_cache");
  DeviceType device_cpu = DeviceType(DEVICE_CPU);
  EXPECT_EQ(GetPersistentCacheDirectory(device_cpu), "/tmp/xla_cache");
  DeviceType device_tpu = DeviceType(DEVICE_TPU);
  EXPECT_EQ(GetPersistentCacheDirectory(device_tpu), "/tmp/xla_cache");
}

}  // namespace
}  // namespace tensorflow
