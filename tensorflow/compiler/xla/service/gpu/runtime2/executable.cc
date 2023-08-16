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
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "third_party/iree/runtime/src/iree/base/api.h"
#include "third_party/iree/runtime/src/iree/hal/api.h"
#include "third_party/iree/runtime/src/iree/hal/drivers/cuda/api.h"
#include "third_party/iree/runtime/src/iree/modules/hal/module.h"
#include "third_party/iree/runtime/src/iree/modules/hal/types.h"
#include "third_party/iree/runtime/src/iree/tooling/numpy_io.h"
#include "third_party/iree/runtime/src/iree/vm/api.h"
#include "third_party/iree/runtime/src/iree/vm/bytecode/module.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/compiler.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/module.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/vm.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/mem.h"
#include "tensorflow/tsl/platform/path.h"

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
  // Bind pre-compiled executable (CUBIN for NVIDIA backend) to runtime input,
  // and return serialized module prepared for passing to the runtime compiler.
  TF_ASSIGN_OR_RETURN(std::string source,
                      BindXlaDeviceKernels(program->debug_options,
                                           *program->module, asm_text, binary));

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

  // Get HLO module name and unique id from the Module operation.
  auto module = *program->module;
  auto unique_id = module->getAttrOfType<mlir::IntegerAttr>("hlo.unique_id");

  return std::unique_ptr<Gpu2RuntimeExecutable>(new Gpu2RuntimeExecutable(
      module.getName().value_or("unknown").str(),
      unique_id ? unique_id.getInt() : 0, std::move(device),
      std::move(bytecode), std::move(program->buffer_sizes),
      std::move(program->debug_options), asm_text, binary, context, instance,
      std::move(modules), function));
}

Gpu2RuntimeExecutable::Gpu2RuntimeExecutable(
    std::string module_name, int32_t module_id,
    std::unique_ptr<HalDevice> device, std::unique_ptr<Bytecode> bytecode,
    std::vector<int64_t> buffer_sizes, DebugOptions debug_options,
    std::string_view asm_text, absl::Span<const uint8_t> binary,
    iree::vm::ref<iree_vm_context_t> context,
    iree::vm::ref<iree_vm_instance_t> instance,
    std::unique_ptr<std::vector<iree_vm_module_t*>> modules,
    iree_vm_function_t function)
    : module_name_(std::move(module_name)),
      module_id_(module_id),
      device_(std::move(device)),
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

static FILE* GetDumpNumpyInputsFile(const DebugOptions& debug_options,
                                    std::string_view module_name,
                                    int32_t module_id) {
  std::string dump_to = debug_options.xla_dump_to();
  if (dump_to.empty()) return nullptr;

  if (std::getenv("XLA_GPU2_IREE_DUMP_NUMPY_INPUTS")) {
    auto file = tsl::io::JoinPath(
        dump_to, FilenameFor(module_id, module_name, "gpu_rt_inputs", "npy"));
    if (FILE* f = fopen(file.c_str(), "w+b")) return f;
    LOG(WARNING) << "Failed to open file for numpy inputs: " << file;
  }
  return nullptr;
}

// Dumps all buffer allocations as numpy arrays into a single `npy` file, which
// allows replaying XLA:GPU invocations with IREE tools.
//
// TODO(ezhulenev): Today this has a very limited usability as IREE tools do not
// support XLA:GPU custom modules, and can only replay XLA:GPU programs that
// have only kernel dispatches. Add an option to replay "runtime IR" with
// builtin XLA tools, somewhat like
// `xla/tools/multihost_hlo_runner:hlo_runner_main` but for runtime IR.
//
// Note: this is a temporary debugging tool while we are prototyping IREEs
// integration into XLA and do not want to touch other parts of XLA.
static void DumpBufferAllocationsAsNumpy(
    const ServiceExecutableRunOptions* run_options,
    iree_allocator_t host_allocator,
    const BufferAllocations& buffer_allocations,
    absl::Span<const int64_t> buffer_sizes, FILE* file) {
  unsigned num_buffer_allocations = buffer_allocations.size();

  se::Stream* stream = run_options->stream();

  iree::vm::ref<iree_hal_allocator_t> allocator;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      iree_make_cstring_view("numpy_dump"), host_allocator, host_allocator,
      &allocator));

  for (unsigned i = 0; i < num_buffer_allocations; ++i) {
    iree_hal_dim_t size = buffer_sizes[i];

    void* ptr = tsl::port::AlignedMalloc(size, IREE_HAL_HEAP_BUFFER_ALIGNMENT);
    auto free_ptr = absl::Cleanup([&]() { tsl::port::AlignedFree(ptr); });

    // Copy buffer from device to a host buffer.
    stream->ThenMemcpy(ptr, buffer_allocations.GetDeviceAddress(i), size);

    // Wrap host buffer into IREE buffer.
    iree::vm::ref<iree_hal_buffer_t> buffer;
    IREE_CHECK_OK(iree_hal_heap_buffer_wrap(
        allocator.get(), IREE_HAL_MEMORY_ACCESS_READ,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_DEFAULT, size,
        iree_make_byte_span(ptr, size),
        {[](void*, iree_hal_buffer_t*) {}, nullptr}, &buffer));

    // Create 1D buffer view from a buffer.
    iree::vm::ref<iree_hal_buffer_view_t> view;
    IREE_CHECK_OK(iree_hal_buffer_view_create(
        buffer.get(), /*shape_rank=*/1, /*shape=*/&size,
        IREE_HAL_ELEMENT_TYPE_INT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        host_allocator, &view));

    // Append host buffer to numpy files.
    iree_numpy_npy_save_options_t opts = IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT;
    auto saved =
        iree_numpy_npy_save_ndarray(file, opts, view.get(), host_allocator);
    if (!iree_status_is_ok(saved))
      LOG(WARNING) << "Failed to save input buffer as numpy array: "
                   << iree::Status(std::move(saved)).ToString();
  }
}

Status Gpu2RuntimeExecutable::Execute(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations,
    const BufferAllocation* temp_alloc) {
  unsigned num_buffer_allocations = buffer_allocations.size();
  CHECK(num_buffer_allocations == buffer_sizes_.size());  // CHECK OK

  iree_allocator_t allocator = iree_allocator_system();

  // Maybe dump all inputs in numpy format for replaying XLA:GPU invocation.
  FILE* dump_inputs_file =
      GetDumpNumpyInputsFile(debug_options_, module_name_, module_id_);
  if (IREE_UNLIKELY(dump_inputs_file)) {
    DumpBufferAllocationsAsNumpy(run_options, allocator, buffer_allocations,
                                 buffer_sizes_, dump_inputs_file);
    fclose(dump_inputs_file);  // ignore errors
  }

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
        buffers.back().get(), /*shape_rank=*/1, /*shape=*/&external_buffer.size,
        IREE_HAL_ELEMENT_TYPE_INT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        allocator, &view));

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
