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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_GEMM_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_GEMM_H_

#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "third_party/iree/runtime/src/iree/hal/api.h"  // IWYU pragma: keep
#include "third_party/iree/runtime/src/iree/vm/api.h"   // IWYU pragma: keep
#include "tensorflow/compiler/xla/service/gpu/runtime2/vm.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// XLA:GPU gemm API custom types
//===-----------------------------------------------------------------------===/

namespace vm {

struct DotDimensionNumbers : public iree::vm::RefObject<DotDimensionNumbers> {
  absl::InlinedVector<int64_t, 4> lhs_batch_dims;
  absl::InlinedVector<int64_t, 4> rhs_batch_dims;
  absl::InlinedVector<int64_t, 4> lhs_contracting_dims;
  absl::InlinedVector<int64_t, 4> rhs_contracting_dims;
};

struct DotPrecision : public iree::vm::RefObject<DotPrecision> {
  absl::InlinedVector<int64_t, 4> precision;
};

struct DotConfig : public iree::vm::RefObject<DotConfig> {
  double alpha_real;
  double alpha_imag;
  double beta;
  int32_t algorithm;
  iree::vm::ref<DotDimensionNumbers> dot_dimension_numbers;
  iree::vm::ref<DotPrecision> dot_precision;
};

}  // namespace vm

//===-----------------------------------------------------------------------===/
// XLA:GPU gemm API
//===-----------------------------------------------------------------------===/

Status DispatchGemm(const vm::ExecutionContext& ctx,
                    iree_hal_allocator_t* device_allocator,
                    iree_hal_buffer_view_t* lhs, iree_hal_buffer_view_t* rhs,
                    iree_hal_buffer_view_t* out, const vm::DotConfig& config);

//===-----------------------------------------------------------------------===/
// XLA:GPU gemm custom module API
//===-----------------------------------------------------------------------===/

namespace vm {

class GemmAPI {
 public:
  explicit GemmAPI(iree_hal_allocator_t* device_allocator);

  // Creates `xla_gpu.dot_dimension_numbers` value.
  iree::StatusOr<iree::vm::ref<vm::DotDimensionNumbers>>
  DotDimensionNumbersCreate(iree::vm::ref<iree_vm_list_t> lhs_batching_dims,
                            iree::vm::ref<iree_vm_list_t> rhs_batching_dims,
                            iree::vm::ref<iree_vm_list_t> lhs_contracting_dims,
                            iree::vm::ref<iree_vm_list_t> rhs_contracting_dims);

  // Creates `xla_gpu.dot_precision` value.
  iree::StatusOr<iree::vm::ref<vm::DotPrecision>> DotPrecisionCreate(
      iree::vm::ref<iree_vm_list_t> precision);

  // Creates `xla_gpu.dot_config` value.
  iree::StatusOr<iree::vm::ref<vm::DotConfig>> DotConfigCreate(
      int32_t algorithm, float alpha_real, float alpha_imag, float beta,
      iree::vm::ref<vm::DotDimensionNumbers> dot_dimension_numbers,
      iree::vm::ref<vm::DotPrecision> dot_precision);

  // Dispatches gemm operation with given buffers and config.
  iree::Status GemmDispatch(iree::vm::ref<ExecutionContext> ctx,
                            iree::vm::ref<iree_hal_buffer_view_t> lhs,
                            iree::vm::ref<iree_hal_buffer_view_t> rhs,
                            iree::vm::ref<iree_hal_buffer_view_t> out,
                            iree::vm::ref<vm::DotConfig> config,
                            iree::vm::ref<Trace> trace);

 private:
  iree_hal_allocator_t* device_allocator_;
};

}  // namespace vm
}  // namespace xla::gpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM
//===----------------------------------------------------------------------===//

IREE_VM_DECLARE_TYPE_ADAPTERS(dot_config, xla::gpu::vm::DotConfig);
IREE_VM_DECLARE_TYPE_ADAPTERS(dot_dimension_numbers,
                              xla::gpu::vm::DotDimensionNumbers);
IREE_VM_DECLARE_TYPE_ADAPTERS(dot_precision, xla::gpu::vm::DotPrecision);

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_GEMM_H_
