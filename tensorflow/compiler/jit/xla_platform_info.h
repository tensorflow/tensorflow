/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_PLATFORM_INFO_H_
#define TENSORFLOW_COMPILER_JIT_XLA_PLATFORM_INFO_H_

#include <memory>
#include <optional>
#include <string>

#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/pjrt_base_device.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "xla/stream_executor/tf_allocator_adapter.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// Holds some information about the platform on which an
// XlaLaunch/_XlaCompile/_XlaRun op must run on. Provides a common layer of
// abstraction for normal, XLA devices and devices inheriting from
// PjRtBaseDevice.
class XlaPlatformInfo {
 public:
  XlaPlatformInfo() : device_type_("") {}
  XlaPlatformInfo(XlaPlatformInfo&&) = default;
  explicit XlaPlatformInfo(
      const DeviceType device_type, se::Platform::Id platform_id,
      const XlaDevice::Metadata* xla_device_metadata,
      const PjRtBaseDevice::Metadata* pjrt_device_metadata,
      std::shared_ptr<se::DeviceMemoryAllocator> device_allocator)
      : device_type_(device_type),
        platform_id_(platform_id),
        xla_device_metadata_(xla_device_metadata),
        pjrt_device_metadata_(pjrt_device_metadata),
        device_allocator_(device_allocator) {}

  XlaPlatformInfo& operator=(XlaPlatformInfo&& other) = default;

  bool UseMultipleStreams() const {
    return xla_device_metadata_ && xla_device_metadata_->UseMultipleStreams();
  }

  // Non-null only when run on an XLA device.
  std::shared_ptr<se::DeviceMemoryAllocator> custom_allocator() const {
    return device_allocator_;
  }

  DeviceType device_type() const { return device_type_; }

  // This is equal to xla_device_metadata()->platform()->id() if
  // xla_device_metadata() is not nullptr.
  se::Platform::Id platform_id() const { return platform_id_; }

  // This may be null if the op this XlaPlatformInfo is for was not placed on an
  // XLA device.
  const XlaDevice::Metadata* xla_device_metadata() const {
    return xla_device_metadata_;
  }
  bool is_on_xla_device() const { return xla_device_metadata() != nullptr; }

  const PjRtBaseDevice::Metadata* pjrt_device_metadata() const {
    return pjrt_device_metadata_;
  }

 private:
  DeviceType device_type_;
  se::Platform::Id platform_id_;

  // xla_device_metadata_ lives in the tensorflow::DeviceBase in which the
  // XlaLaunch/_XlaCompile/_XlaRun op is placed and thus does not die before the
  // XlaLaunch/_XlaCompile/_XlaRun OpKernel.
  const XlaDevice::Metadata* xla_device_metadata_;

  // pjrt_device_metadata_ lives in tensorflow::PjRtBaseDevice in which the
  // XlaLaunch/XlaCompileOnDemand op is placed and thus does not die before the
  // op kernel.
  const PjRtBaseDevice::Metadata* pjrt_device_metadata_;

  // If the op associated with this XlaPlatformInfo is placed on an XLA device
  // then device_allocator_ is the xla::Backend's memory allocator.  If the op
  // is placed on a regular CPU or GPU device then device_allocator_ is null.
  // The allocator is of unknown provenance; keep it in a shared pointer to
  // set an artificial refcount of one.
  std::shared_ptr<se::DeviceMemoryAllocator> device_allocator_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaPlatformInfo);
};

// Returns a set containing the device ids contained in visible_device_list or
// nullopt if it is empty. It returns error in case of malformed configuration
// string.
StatusOr<std::optional<std::set<int>>> ParseVisibleDeviceList(
    absl::string_view visible_device_list);

// Builds a DeviceCompiler that uses xla::LocalClient using `platform_info` and
// sets *xla_device_compiler to point to it. Uses flags from
// `MarkForCompilationPassFlags` for configuring the persistor used in the
// DeviceCompiler.
Status BuildXlaDeviceCompiler(
    DeviceBase* dev, FunctionLibraryRuntime* flr,
    const XlaPlatformInfo& platform_info,
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>**
        xla_device_compiler);

// Fetches a DeviceCompiler from the tfrt_global resource manager (or creates
// one there if not found) that uses xla::PjRtClient using an appropriate
// PjRtClient for `platform_info.device_type()` and sets *pjrt_device_compiler
// to point to it. Also fetches/creates a DeviceCompilationProfiler from/in the
// tfrt_global resource manager for `platform_info.device_type()` and sets
// *profiler to point to it.  Uses flags from `MarkForCompilationPassFlags` for
// configuring the persistor used in the DeviceCompiler. Please note that
// non-XLA devices aren't supported yet. This is because:
// 1. PjRtClient doesn't support data transfer for non-XLA devices yet
// 2. Fetching the PjRtClient for non-XLA devices is also not supported yet
Status GetOrCreatePjRtDeviceCompilerAndProfiler(
    const OpKernelContext& ctx, const XlaPlatformInfo& platform_info,
    FunctionLibraryRuntime* flr,
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>**
        pjrt_device_compiler,
    DeviceCompilationProfiler** profiler);

// Same as the above function but takes the resource manager `rm` instead of an
// OpKernelContext.
Status GetOrCreatePjRtDeviceCompilerAndProfiler(
    const XlaPlatformInfo& platform_info, ResourceMgr* rm,
    FunctionLibraryRuntime* flr,
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>**
        pjrt_device_compiler,
    DeviceCompilationProfiler** profiler);

// Returns information about the platform from kernel context.
XlaPlatformInfo XlaPlatformInfoFromDevice(DeviceBase* device);

// Obtains persistent cache directory for executables that target a given device
// based off xla flags. If you shouldn't use persistent caching, returns "".
std::string GetPersistentCacheDirectory(
    const DeviceType& compilation_device_type);

// Returns allocator from platform info if non-null, or populate and return a
// pointer to the allocator adapter with allocator from context.
//
// This is necessary because for XLA devices the underlying TF allocator returns
// dummy tensors.
//
// `stream` parameter is nullable when running on host.
std::shared_ptr<se::DeviceMemoryAllocator> GetAllocator(
    DeviceBase* device, se::Stream* stream,
    const XlaPlatformInfo& platform_info);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_PLATFORM_INFO_H_
