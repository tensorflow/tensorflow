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

#include "tensorflow/compiler/xla/service/gpu/runtime2/executable.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "third_party/iree/runtime/src/iree/base/api.h"
#include "third_party/iree/runtime/src/iree/hal/api.h"
#include "third_party/iree/runtime/src/iree/hal/drivers/cuda/api.h"
#include "third_party/iree/runtime/src/iree/modules/hal/module.h"
#include "third_party/iree/runtime/src/iree/modules/hal/types.h"
#include "third_party/iree/runtime/src/iree/vm/api.h"
#include "third_party/iree/runtime/src/iree/vm/bytecode/module.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/compiler.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/module.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/vm.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla::gpu {

// TODO(ezhulenev): In this file we need to remove all IREE_CHECK_OK and replace
// with RETURN_IF_ERROR macro that will do iree_status_t => Status conversion.

//===-----------------------------------------------------------------------===/
// HalDevice
//===-----------------------------------------------------------------------===/

struct HalDevice {
  ~HalDevice() {
    iree_hal_device_destroy(device);
    iree_hal_driver_destroy(driver);
  }

  iree_hal_driver_t* driver = nullptr;
  iree_hal_device_t* device = nullptr;
};

static iree_status_t CreateCudaDriver(iree_allocator_t allocator,
                                      HalDevice* device) {
  iree_string_view_t driver_name = iree_make_cstring_view("cuda");

  iree_hal_cuda_device_params_t default_params;
  iree_hal_cuda_device_params_initialize(&default_params);
  default_params.command_buffer_mode = IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM;
  default_params.allow_inline_execution = false;

  iree_hal_cuda_driver_options_t driver_options;
  iree_hal_cuda_driver_options_initialize(&driver_options);
  driver_options.default_device_index = 0;

  IREE_CHECK_OK(iree_hal_cuda_driver_create(driver_name, &default_params,
                                            &driver_options, allocator,
                                            &device->driver));

  return iree_ok_status();
}

static iree_status_t CreateCudaDevice(iree_allocator_t allocator,
                                      HalDevice* device) {
  IREE_CHECK_OK(iree_hal_driver_create_device_by_id(
      device->driver, /*device_id=*/0,
      /*param_count=*/0, /*params=*/nullptr, allocator, &device->device));
  return iree_ok_status();
}

//===-----------------------------------------------------------------------===/
// Gpu2RuntimeExecutable
//===-----------------------------------------------------------------------===/

// TODO(ezhulenev): Crashing is absolutely not ok here, add proper error
// handling and remove CHECK and IREE_CHECK_OK.

/*static*/ StatusOr<std::unique_ptr<Gpu2RuntimeExecutable>>
Gpu2RuntimeExecutable::Create(std::unique_ptr<Gpu2RuntimeProgram> program,
                              std::string_view asm_text,
                              const std::vector<uint8_t>& binary) {
  CHECK_OK(BindXlaDeviceKernels(*program->module, asm_text, binary));
  auto source = llvm_ir::DumpToString(*program->module);

  // Compile IR in IREE input dialect(s) into IREE VM flatbuffer.
  auto compiler = CreateRuntimeCompiler();
  CHECK(compiler->SetFlag("--iree-hal-target-backends=cuda"));
  CHECK(compiler->ParseSourceBuffer(source));
  auto bytecode = compiler->CompileStandardPipeline();
  CHECK(bytecode);

  // TODO(ezhulenev): We need a better strategy for managing IREE resources: VM
  // instances, contexts, devices, etc. What should be shared between all XLA
  // executables, and what should be unique to each executable?
  iree_allocator_t allocator = iree_allocator_system();

  // Create the root isolated VM instance that we can create contexts within.
  iree::vm::ref<iree_vm_instance_t> instance;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        allocator, &instance));

  // TODO(ezhulenev): CUDA devices/drivers should be created globally, and share
  // CUDA context with corresponding StreamExecutor.
  auto device = std::make_unique<HalDevice>();
  IREE_CHECK_OK(CreateCudaDriver(allocator, device.get()));
  IREE_CHECK_OK(CreateCudaDevice(allocator, device.get()));

  auto modules = std::make_unique<std::vector<iree_vm_module_t*>>();

  // Load HAL module.
  IREE_CHECK_OK(iree_hal_module_register_all_types(instance.get()));
  IREE_CHECK_OK(iree_hal_module_create(instance.get(), device->device,
                                       IREE_HAL_MODULE_FLAG_NONE, allocator,
                                       &modules->emplace_back()));

  // Load XLA:GPU module.
  IREE_CHECK_OK(RegisterXlaGpuTypes(instance.get()));
  IREE_CHECK_OK(CreateXlaGpuModule(instance.get(), allocator,
                                   iree_hal_device_allocator(device->device),
                                   &modules->emplace_back()));

  // Load module compiled from XLA program to a VM flatbuffer.
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      instance.get(),
      iree_make_const_byte_span(bytecode->data(), bytecode->lenth()),
      /*archive_allocator=*/iree_allocator_null(), allocator,
      &modules->emplace_back()));

  // TODO(ezhulenev): Figure out what is the correct context management strategy
  // given that executable can be executed concurrently. This is almost
  // certainly wrong, and will lead to data races.
  iree::vm::ref<iree_vm_context_t> context;
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      instance.get(), IREE_VM_CONTEXT_FLAG_NONE, modules->size(),
      modules->data(), allocator, &context));

  // Fully qualified entry point name.
  auto module_name = iree_vm_module_name(modules->back());
  std::string qualified_name = std::string(module_name.data, module_name.size) +
                               "." + program->entry_point;

  // Look up the function by fully-qualified name (module.func).
  iree_vm_function_t function;
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      context.get(), iree_make_cstring_view(qualified_name.c_str()),
      &function));

  return std::unique_ptr<Gpu2RuntimeExecutable>(new Gpu2RuntimeExecutable(
      std::move(device), std::move(bytecode), std::move(program->buffer_sizes),
      std::move(program->debug_options), asm_text, binary, context, instance,
      std::move(modules), function));
}

Gpu2RuntimeExecutable::Gpu2RuntimeExecutable(
    std::unique_ptr<HalDevice> device, std::unique_ptr<Bytecode> bytecode,
    std::vector<int64_t> buffer_sizes, DebugOptions debug_options,
    std::string_view asm_text, absl::Span<const uint8_t> binary,
    iree::vm::ref<iree_vm_context_t> context,
    iree::vm::ref<iree_vm_instance_t> instance,
    std::unique_ptr<std::vector<iree_vm_module_t*>> modules,
    iree_vm_function_t function)
    : device_(std::move(device)),
      bytecode_(std::move(bytecode)),
      buffer_sizes_(std::move(buffer_sizes)),
      debug_options_(std::move(debug_options)),
      asm_text_(asm_text),
      binary_(binary),
      context_(std::move(context)),
      instance_(std::move(instance)),
      modules_(std::move(modules)),
      function_(std::move(function)) {
  auto name = iree_vm_function_name(&function_);
  VLOG(1) << "Created XLA:GPU executable: function name = "
          << std::string_view(name.data, name.size);
}

Gpu2RuntimeExecutable::~Gpu2RuntimeExecutable() {
  for (auto module : *modules_) iree_vm_module_release(module);
}

Status Gpu2RuntimeExecutable::Execute(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations,
    const BufferAllocation* temp_alloc) {
  unsigned num_buffer_allocations = buffer_allocations.size();
  CHECK(num_buffer_allocations == buffer_sizes_.size());  // CHECK OK

  iree_allocator_t allocator = iree_allocator_system();

  // Prepare a list for passing arguments to the function.
  iree::vm::ref<iree_vm_list_t> inputs;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                    buffer_allocations.size(), allocator,
                                    &inputs));

  // Add execution context as the first arguments.
  auto execution_context = iree::vm::make_ref<vm::ExecutionContext>(
      run_options, &debug_options_,
      vm::ExecutionContext::ExecutableSource{asm_text_, binary_});

  // TODO(ezhulenev): Can we do ref_move here?
  IREE_CHECK_OK(iree_vm_list_push_ref_retain(inputs.get(), execution_context));

  // Import argument buffers as device-local IREE buffers.
  std::vector<iree::vm::ref<iree_hal_buffer_t>> buffers;

  // Convert XLA buffer allocations to IREE buffer views.
  for (unsigned i = 0; i < num_buffer_allocations; ++i) {
    // Import XLA buffer as an IREE external buffer.
    iree_hal_external_buffer_t external_buffer;
    external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
    external_buffer.flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE;
    external_buffer.size = buffer_sizes_[i];
    external_buffer.handle.device_allocation.ptr = reinterpret_cast<uint64_t>(
        buffer_allocations.GetDeviceAddress(i).opaque());

    // All XLA:GPU buffer arguments are always allocated on device.
    iree_hal_buffer_params_t buffer_params = {
        /*usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
        /*access=*/IREE_HAL_MEMORY_ACCESS_ALL,
        /*type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    };

    // Import XLA buffer with no-op release callback, because the lifetime of
    // arguments is managed by XLA itself and we get non-owning pointers.
    IREE_CHECK_OK(iree_hal_allocator_import_buffer(
        iree_hal_device_allocator(device_->device), buffer_params,
        &external_buffer, {[](void*, iree_hal_buffer_t*) {}, nullptr},
        &buffers.emplace_back()));

    // In XLA all buffer arguments are vectors of i8 data type.
    iree_hal_buffer_view_t* view = nullptr;
    IREE_CHECK_OK(iree_hal_buffer_view_create(
        buffers.back().get(),
        /*shape_rank=*/1,
        /*shape=*/&external_buffer.size, IREE_HAL_ELEMENT_TYPE_INT_8,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, allocator, &view));

    // Move buffer view to the inputs list.
    iree_vm_ref_t view_ref = iree_hal_buffer_view_move_ref(view);
    IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs.get(), &view_ref));
  }

  IREE_CHECK_OK(iree_vm_invoke(context_.get(), function_,
                               IREE_VM_INVOCATION_FLAG_NONE,
                               /*policy=*/nullptr, inputs.get(),
                               /*outputs=*/nullptr, allocator));

  return OkStatus();
}

}  // namespace xla::gpu
