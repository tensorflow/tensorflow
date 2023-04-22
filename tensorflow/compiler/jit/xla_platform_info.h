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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace tensorflow {

// Holds some information about the platform on which an
// XlaLaunch/_XlaCompile/_XlaRun op must run on. Provides a common layer of
// abstraction for normal and XLA devices.
class XlaPlatformInfo {
 public:
  XlaPlatformInfo() : device_type_("") {}
  XlaPlatformInfo(XlaPlatformInfo&&) = default;
  explicit XlaPlatformInfo(
      const DeviceType device_type, se::Platform::Id platform_id,
      const XlaDevice::Metadata* xla_device_metadata,
      std::shared_ptr<se::DeviceMemoryAllocator> device_allocator)
      : device_type_(device_type),
        platform_id_(platform_id),
        xla_device_metadata_(xla_device_metadata),
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

 private:
  DeviceType device_type_;
  se::Platform::Id platform_id_;

  // xla_device_metadata_ lives in the tensorflow::DeviceBase in which the
  // XlaLaunch/_XlaCompile/_XlaRun op is placed and thus does not die before the
  // XlaLaunch/_XlaCompile/_XlaRun OpKernel.
  const XlaDevice::Metadata* xla_device_metadata_;

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
StatusOr<absl::optional<std::set<int>>> ParseVisibleDeviceList(
    absl::string_view visible_device_list);

// Returns created XLA compilation cache.
Status BuildXlaCompilationCache(DeviceBase* dev, FunctionLibraryRuntime* flr,
                                const XlaPlatformInfo& platform_info,
                                XlaCompilationCache** cache);

// Returns information about the platform from kernel context.
XlaPlatformInfo XlaPlatformInfoFromDevice(DeviceBase* device);

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

// Returns created options for the XLA compiler, and writes the used allocator
// into `tf_allocator_adapter`.
XlaCompiler::Options GenerateCompilerOptions(
    const XlaCompilationCache& cache,
    const FunctionLibraryRuntime& function_library, DeviceBase* device,
    se::Stream* stream, const XlaPlatformInfo& platform_info,
    bool has_ref_vars);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_PLATFORM_INFO_H_
