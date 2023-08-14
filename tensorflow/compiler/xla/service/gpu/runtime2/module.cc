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

#include "tensorflow/compiler/xla/service/gpu/runtime2/module.h"

#include <memory>

#include "third_party/iree/runtime/src/iree/base/api.h"  // IWYU pragma: keep
#include "third_party/iree/runtime/src/iree/hal/api.h"   // IWYU pragma: keep
#include "third_party/iree/runtime/src/iree/modules/hal/types.h"  // IWYU pragma: keep
#include "third_party/iree/runtime/src/iree/vm/api.h"  // IWYU pragma: keep
#include "third_party/iree/runtime/src/iree/vm/native_module_cc.h"
#include "third_party/iree/runtime/src/iree/vm/native_module_packing.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/gemm.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/graph.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/kernel.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/memcpy.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/vm.h"

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// XLA:GPU custom module state
//===-----------------------------------------------------------------------===/

using vm::GemmAPI;
using vm::GraphAPI;
using vm::KernelAPI;
using vm::MemcpyAPI;
using vm::TraceAPI;

class XlaGpuModuleState : public GemmAPI,
                          public KernelAPI,
                          public MemcpyAPI,
                          public GraphAPI,
                          public TraceAPI {
 public:
  explicit XlaGpuModuleState(iree_hal_allocator_t* device_allocator)
      : GemmAPI(device_allocator),
        KernelAPI(device_allocator),
        MemcpyAPI(device_allocator),
        GraphAPI(device_allocator) {}
};

//===----------------------------------------------------------------------===//
// Helper functions for exporting native functions to a module
//===----------------------------------------------------------------------===//

// Casts a pointer to a member function of a base class to a pointer to a member
// function of the parent class. ("Pointer to member conversions" [conv.mem]).
template <typename Base, typename... Params>
static constexpr auto UpCast(iree::Status (Base::*fn)(Params...)) {
  return static_cast<iree::Status (XlaGpuModuleState::*)(Params...)>(fn);
}

template <typename Base, typename R, typename... Params>
static constexpr auto UpCast(iree::StatusOr<R> (Base::*fn)(Params...)) {
  return static_cast<iree::StatusOr<R> (XlaGpuModuleState::*)(Params...)>(fn);
}

template <typename Base, typename... Params>
static constexpr auto MakeApiFunction(const char* name,
                                      iree::Status (Base::*fn)(Params...)) {
  return iree::vm::MakeNativeFunction(name, UpCast(fn));
}

template <typename Base, typename R, typename... Params>
static constexpr auto MakeApiFunction(
    const char* name, iree::StatusOr<R> (Base::*fn)(Params...)) {
  return iree::vm::MakeNativeFunction(name, UpCast(fn));
}

//===-----------------------------------------------------------------------===/
// XLA:GPU custom module
//===-----------------------------------------------------------------------===/

static const iree::vm::NativeFunction<XlaGpuModuleState> kXlaGpuFunctions[] = {
    // XLA:GPU Gemm APIs
    MakeApiFunction("dot_dimension_numbers.create",
                    &GemmAPI::DotDimensionNumbersCreate),
    MakeApiFunction("dot_precision.create", &GemmAPI::DotPrecisionCreate),
    MakeApiFunction("dot_config.create", &GemmAPI::DotConfigCreate),
    MakeApiFunction("gemm.dispatch", &GemmAPI::GemmDispatch),

    // XLA:GPU kernel APIs
    MakeApiFunction("kernel.create", &KernelAPI::KernelCreate),
    MakeApiFunction("kernel.dispatch", &KernelAPI::KernelDispatch),

    // XLA:GPU memcpy APIs
    MakeApiFunction("memcpy.d2d", &MemcpyAPI::MemcpyD2D),
    MakeApiFunction("memcpy.load.i1", &MemcpyAPI::LoadI1),

    // XLA:GPU graph APIs.
    MakeApiFunction("graph.create", &GraphAPI::GraphCreate),
    MakeApiFunction("graph.kernel_node.create",
                    &GraphAPI::GraphKernelNodeCreate),
    MakeApiFunction("graph.memcpy_node.d2d.create",
                    &GraphAPI::GraphMemcpyD2DNodeCreate),
    MakeApiFunction("graph.execute", &GraphAPI::GraphExecute),

    // XLA:GPU tracing APIs
    MakeApiFunction("trace.create", &TraceAPI::TraceCreate),
};

class XlaGpuModule : public iree::vm::NativeModule<XlaGpuModuleState> {
  using NativeModule = iree::vm::NativeModule<XlaGpuModuleState>;

 public:
  XlaGpuModule(iree_vm_instance_t* instance, iree_allocator_t host_allocator,
               iree_hal_allocator_t* device_allocator);

  iree::StatusOr<std::unique_ptr<XlaGpuModuleState>> CreateState(
      iree_allocator_t host_allocator) override;

 private:
  static constexpr uint32_t kVersion = 0;
  iree_hal_allocator_t* device_allocator_;
};

XlaGpuModule::XlaGpuModule(iree_vm_instance_t* instance,
                           iree_allocator_t host_allocator,
                           iree_hal_allocator_t* device_allocator)
    : NativeModule("xla_gpu", kVersion, instance, host_allocator,
                   kXlaGpuFunctions),
      device_allocator_(device_allocator) {}

iree::StatusOr<std::unique_ptr<XlaGpuModuleState>> XlaGpuModule::CreateState(
    iree_allocator_t host_allocator) {
  return std::make_unique<XlaGpuModuleState>(device_allocator_);
}

//===-----------------------------------------------------------------------===/
// XLA:GPU custom module constructor
//===-----------------------------------------------------------------------===/

iree_status_t CreateXlaGpuModule(iree_vm_instance_t* instance,
                                 iree_allocator_t host_allocator,
                                 iree_hal_allocator_t* device_allocator,
                                 iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);

  RegisterXlaGpuTypes(instance);

  auto module = std::make_unique<XlaGpuModule>(instance, host_allocator,
                                               device_allocator);
  *out_module = module.release()->interface();

  return iree_ok_status();
}

//===-----------------------------------------------------------------------===/
// XLA:GPU custom module type registration
//===-----------------------------------------------------------------------===/

template <typename T>
static iree_status_t RegisterType(iree_vm_instance_t* instance,
                                  const char* name, iree_vm_ref_type_t* out) {
  static iree_vm_ref_type_descriptor_t descriptor = {nullptr};

  descriptor.type_name = iree_make_cstring_view(name);
  descriptor.offsetof_counter = T::offsetof_counter();
  descriptor.destroy = T::DirectDestroy;

  return iree_vm_instance_register_type(instance, &descriptor, out);
}

iree_status_t RegisterXlaGpuTypes(iree_vm_instance_t* instance) {
  // XLA:GPU Execution context type
  IREE_RETURN_IF_ERROR(RegisterType<vm::ExecutionContext>(
      instance, "xla_gpu.execution_context", &execution_context_registration));

  // XLA:GPU Gemm types
  IREE_RETURN_IF_ERROR(RegisterType<vm::DotDimensionNumbers>(
      instance, "xla_gpu.dot_dimension_numbers",
      &dot_dimension_numbers_registration));
  IREE_RETURN_IF_ERROR(RegisterType<vm::DotPrecision>(
      instance, "xla_gpu.dot_precision", &dot_precision_registration));
  IREE_RETURN_IF_ERROR(RegisterType<vm::DotConfig>(
      instance, "xla_gpu.dot_config", &dot_config_registration));

  // XLA:GPU kernel dispatch types
  IREE_RETURN_IF_ERROR(RegisterType<vm::Kernel>(instance, "xla_gpu.kernel",
                                                &kernel_registration));

  // XLA:GPU graph types
  IREE_RETURN_IF_ERROR(
      RegisterType<vm::Graph>(instance, "xla_gpu.graph", &graph_registration));
  IREE_RETURN_IF_ERROR(RegisterType<vm::GraphNode>(
      instance, "xla_gpu.graph.node", &graph_node_registration));

  // XLA:GPU tracing types
  IREE_RETURN_IF_ERROR(
      RegisterType<vm::Trace>(instance, "xla_gpu.trace", &trace_registration));

  return iree_ok_status();
}

}  // namespace xla::gpu
