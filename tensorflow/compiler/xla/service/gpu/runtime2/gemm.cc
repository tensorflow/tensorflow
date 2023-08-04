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

#include "tensorflow/compiler/xla/service/gpu/runtime2/gemm.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/hal.h"
#include "tensorflow/compiler/xla/service/gpu/runtime2/vm.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

using tsl::profiler::ScopedAnnotation;

//===-----------------------------------------------------------------------===/
// XLA:GPU gemm API
//===-----------------------------------------------------------------------===/

static StatusOr<GemmConfig> GetGemmConfig(iree_hal_buffer_view_t* lhs,
                                          iree_hal_buffer_view_t* rhs,
                                          iree_hal_buffer_view_t* out,
                                          const vm::DotConfig& config) {
  int64_t compute_precision =
      config.dot_precision->precision.empty()
          ? se::blas::kDefaultComputePrecision
          : *absl::c_max_element(config.dot_precision->precision);

  return GemmConfig::For(
      GetBufferShape(lhs), config.dot_dimension_numbers->lhs_batch_dims,
      config.dot_dimension_numbers->lhs_contracting_dims,  // lhs
      GetBufferShape(rhs), config.dot_dimension_numbers->rhs_batch_dims,
      config.dot_dimension_numbers->rhs_contracting_dims,  // rhs
      GetBufferShape(out),                                 // out
      config.alpha_real, config.alpha_imag, config.beta, config.algorithm,
      compute_precision);
}

Status DispatchGemm(const vm::ExecutionContext& ctx,
                    iree_hal_allocator_t* device_allocator,
                    iree_hal_buffer_view_t* lhs, iree_hal_buffer_view_t* rhs,
                    iree_hal_buffer_view_t* out, const vm::DotConfig& config) {
  se::Stream* stream = ctx.run_options->stream();

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase lhs_data,
                      GetDeviceMemory(device_allocator, lhs));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase rhs_data,
                      GetDeviceMemory(device_allocator, rhs));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase output_data,
                      GetDeviceMemory(device_allocator, out));

  bool deterministic = ctx.debug_options->xla_gpu_deterministic_ops();

  TF_ASSIGN_OR_RETURN(auto gemm_config, GetGemmConfig(lhs, rhs, out, config));
  return RunGemm(gemm_config, lhs_data, rhs_data, output_data, deterministic,
                 stream);
}

//===-----------------------------------------------------------------------===/
// XLA:GPU gemm custom module API
//===-----------------------------------------------------------------------===/

namespace vm {

GemmAPI::GemmAPI(iree_hal_allocator_t* device_allocator)
    : device_allocator_(device_allocator) {}

iree::StatusOr<iree::vm::ref<vm::DotDimensionNumbers>>
GemmAPI::DotDimensionNumbersCreate(
    iree::vm::ref<iree_vm_list_t> lhs_batching_dims,
    iree::vm::ref<iree_vm_list_t> rhs_batching_dims,
    iree::vm::ref<iree_vm_list_t> lhs_contracting_dims,
    iree::vm::ref<iree_vm_list_t> rhs_contracting_dims) {
  auto ref = iree::vm::make_ref<vm::DotDimensionNumbers>();

  IREE_ASSIGN_OR_RETURN(ref->lhs_batch_dims,
                        vm::GetI64Vector(lhs_batching_dims.get()));
  IREE_ASSIGN_OR_RETURN(ref->rhs_batch_dims,
                        vm::GetI64Vector(rhs_batching_dims.get()));
  IREE_ASSIGN_OR_RETURN(ref->lhs_contracting_dims,
                        vm::GetI64Vector(lhs_contracting_dims.get()));
  IREE_ASSIGN_OR_RETURN(ref->rhs_contracting_dims,
                        vm::GetI64Vector(rhs_contracting_dims.get()));

  return ref;
}

iree::StatusOr<iree::vm::ref<vm::DotPrecision>> GemmAPI::DotPrecisionCreate(
    iree::vm::ref<iree_vm_list_t> precision) {
  auto ref = iree::vm::make_ref<vm::DotPrecision>();
  IREE_ASSIGN_OR_RETURN(ref->precision, vm::GetI64Vector(precision.get()));
  return ref;
}

iree::StatusOr<iree::vm::ref<vm::DotConfig>> GemmAPI::DotConfigCreate(
    int32_t algorithm, float alpha_real, float alpha_imag, float beta,
    iree::vm::ref<vm::DotDimensionNumbers> dot_dimension_numbers,
    iree::vm::ref<vm::DotPrecision> dot_precision) {
  auto ref = iree::vm::make_ref<vm::DotConfig>();
  ref->algorithm = algorithm;
  ref->alpha_real = alpha_real;
  ref->alpha_imag = alpha_imag;
  ref->beta = beta;
  ref->dot_dimension_numbers = std::move(dot_dimension_numbers);
  ref->dot_precision = std::move(dot_precision);
  return ref;
}

iree::Status GemmAPI::GemmDispatch(iree::vm::ref<ExecutionContext> ctx,
                                   iree::vm::ref<iree_hal_buffer_view_t> lhs,
                                   iree::vm::ref<iree_hal_buffer_view_t> rhs,
                                   iree::vm::ref<iree_hal_buffer_view_t> out,
                                   iree::vm::ref<vm::DotConfig> config,
                                   iree::vm::ref<Trace> trace) {
  ScopedAnnotation annotation([&] { return ToScopedAnnotationName(*trace); });
  return FromStatus(DispatchGemm(*ctx, device_allocator_, lhs.get(), rhs.get(),
                                 out.get(), *config));
}

}  // namespace vm
}  // namespace xla::gpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(dot_config, xla::gpu::vm::DotConfig);
IREE_VM_DEFINE_TYPE_ADAPTERS(dot_dimension_numbers,
                             xla::gpu::vm::DotDimensionNumbers);
IREE_VM_DEFINE_TYPE_ADAPTERS(dot_precision, xla::gpu::vm::DotPrecision);
