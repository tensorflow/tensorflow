/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"

#include <functional>
#include <memory>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace tensorflow {

const char* const DEVICE_CPU_XLA_JIT = "XLA_CPU_JIT";
const char* const DEVICE_GPU_XLA_JIT = "XLA_GPU_JIT";

// The XlaCompilationAllocator doesn't actually back any Tensors with storage
// buffers of values: instead for each Tensor it stores a
// XlaExpression which corresponds to the XLA computation
// represented by the Tensor.
class XlaCompilationAllocator : public Allocator {
 public:
  XlaCompilationAllocator() {}
  ~XlaCompilationAllocator() override {}

  string Name() override { return "tla_jit"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    // Regardless of the size requested, always allocates an XlaExpression.
    // Respects the aligment request because there is alignment checking even
    // for Tensors whose data is never accessed.
    void* p = port::AlignedMalloc(sizeof(XlaExpression), alignment);
    XlaExpression* expression = reinterpret_cast<XlaExpression*>(p);
    new (expression) XlaExpression();
    return expression;
  }

  void DeallocateRaw(void* ptr) override {
    XlaExpression* expression = reinterpret_cast<XlaExpression*>(ptr);
    expression->~XlaExpression();
    port::AlignedFree(ptr);
  }

  // Make sure that even tensors with 0 elements have allocated
  // buffers, so they get ids to track.
  bool ShouldAllocateEmptyTensors() override { return true; }

  void GetStats(AllocatorStats* stats) override { stats->Clear(); }

 private:
  // Don't run any constructors or destructors for complex objects,
  // since there is no backing store for the tensor to run them
  // on. strings are the only complex objects currently stored in
  // Tensors. If others are added, this set of overrides must be
  // extended to include them.
  void RunStringCtor(string* p, size_t n) override {}
  void RunStringDtor(string* p, size_t n) override {}
  void RunResourceCtor(ResourceHandle* p, size_t n) override {}
  void RunResourceDtor(ResourceHandle* p, size_t n) override {}
};

XlaCompilationDevice::XlaCompilationDevice(const SessionOptions& options,
                                           DeviceType type)
    : LocalDevice(options,
                  Device::BuildDeviceAttributes(
                      "", type, Bytes(256 << 20), DeviceLocality(),
                      strings::StrCat("device: XLA JIT device ", type.type())),
                  cpu_allocator()),
      allocator_(new XlaCompilationAllocator()) {}

XlaCompilationDevice::~XlaCompilationDevice() {}

Allocator* XlaCompilationDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_.get();
}

Status XlaCompilationDevice::Sync() { return Status::OK(); }

Status XlaCompilationDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  return errors::InvalidArgument(
      "Tla JIT Device should not parse tensor from proto");
}

// Is platform 'id' supported by XLA?
static bool IsPlatformSupported(perftools::gputools::Platform::Id id) {
  auto platform = perftools::gputools::MultiPlatformManager::PlatformWithId(id);
  if (!platform.ok()) return false;
  return xla::ClientLibrary::GetOrCreateLocalClient(platform.ValueOrDie()).ok();
}

XlaOpRegistry::XlaOpRegistry() = default;
XlaOpRegistry::~XlaOpRegistry() = default;

/* static */ void XlaOpRegistry::RegisterJitDevice(
    const string& device_name, const string& jit_device_name, bool requires_jit,
    bool enable_jit_by_default) {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  auto result = registry.jit_devices_.emplace(
      device_name,
      JitDevice{jit_device_name, requires_jit, enable_jit_by_default});
  CHECK(result.second ||
        result.first->second.jit_device_name == jit_device_name);
}

/* static */ bool XlaOpRegistry::GetJitDevice(const string& device_name,
                                              const string** jit_device_name,
                                              bool* requires_jit,
                                              bool* enable_jit_by_default) {
  XlaOpRegistry& registry = Instance();

  // Lazily register the CPU and GPU JIT devices the first time GetJitDevice is
  // called.
  static void* registration = [&registry]() {
    mutex_lock lock(registry.mutex_);
    if (IsPlatformSupported(perftools::gputools::host::kHostPlatformId)) {
      registry.jit_devices_[DEVICE_CPU] = {DEVICE_CPU_XLA_JIT, false, false};
    }
    if (IsPlatformSupported(perftools::gputools::cuda::kCudaPlatformId)) {
      registry.jit_devices_[DEVICE_GPU] = {DEVICE_GPU_XLA_JIT, false, true};
    }
    return nullptr;
  }();
  (void)registration;

  mutex_lock lock(registry.mutex_);
  auto it = registry.jit_devices_.find(device_name);
  if (it == registry.jit_devices_.end()) return false;
  if (jit_device_name) *jit_device_name = &it->second.jit_device_name;
  if (requires_jit) *requires_jit = it->second.requires_jit;
  if (enable_jit_by_default) {
    *enable_jit_by_default = it->second.enable_jit_by_default;
  }
  return true;
}

void XlaOpRegistry::RegisterJitKernels() {
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);

  if (registry.jit_kernels_registered_) return;
  registry.jit_kernels_registered_ = true;

  for (const auto& entry : registry.kernels_) {
    for (const XlaKernel& k : entry.second) {
      auto it = registry.ops_.find(k.kernel_def->op());
      CHECK(it != registry.ops_.end())
          << "Missing XLA op registration for op " << k.kernel_def->op();
      registry.kernel_registrars_.emplace_back(
          new kernel_factory::OpKernelRegistrar(new KernelDef(*k.kernel_def),
                                                "XlaJitOp", it->second));
    }
  }
}

std::vector<const KernelDef*> XlaOpRegistry::DeviceKernels(
    const string& jit_device_type) {
  std::vector<const KernelDef*> kernels;
  XlaOpRegistry& registry = Instance();
  mutex_lock lock(registry.mutex_);
  for (const XlaKernel& k : registry.kernels_.at(jit_device_type)) {
    if (!k.jit_only) {
      kernels.push_back(k.kernel_def.get());
    }
  }
  return kernels;
}

XlaOpRegistry& XlaOpRegistry::Instance() {
  static XlaOpRegistry* r = new XlaOpRegistry;
  return *r;
}

XlaOpRegistrar::XlaOpRegistrar(StringPiece name,
                               XlaOpRegistry::Factory factory) {
  XlaOpRegistry& registry = XlaOpRegistry::Instance();
  mutex_lock lock(registry.mutex_);
  CHECK(registry.ops_.emplace(name.ToString(), factory).second)
      << "Duplicate XLA op registration " << name;
}

XlaKernelRegistrar::XlaKernelRegistrar(bool jit_only, const KernelDef* def) {
  XlaOpRegistry& registry = XlaOpRegistry::Instance();
  mutex_lock lock(registry.mutex_);
  registry.kernels_[def->device_type()].push_back(XlaOpRegistry::XlaKernel{
      jit_only, std::unique_ptr<const KernelDef>(def)});
}

}  // namespace tensorflow
