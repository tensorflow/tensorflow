/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATMUL_ACL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATMUL_ACL_H_

#include <iostream>

#include "tensorflow/core/platform/types.h"

#ifdef XLA_CPU_USE_ACL
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/Utils.h"

struct acl_matmul_obj_t {
  arm_compute::NEGEMM gemm;
  arm_compute::NETranspose trans_lhs;
  arm_compute::NETranspose trans_rhs;
  arm_compute::Tensor rhs_tensor;
  arm_compute::Tensor rhs_acc_tensor;
  arm_compute::Tensor lhs_tensor;
  arm_compute::Tensor lhs_acc_tensor;
  arm_compute::Tensor out_tensor;
};

struct acl_matmul_conf_t {
  bool with_bias;
  bool is_trans_lhs;
  bool is_trans_rhs;
  arm_compute::TensorInfo lhs_info;
  arm_compute::TensorInfo lhs_acc_info;
  arm_compute::TensorInfo rhs_info;
  arm_compute::TensorInfo rhs_acc_info;
  arm_compute::TensorInfo out_info;
  arm_compute::GEMMInfo gemm_info;
  float alpha;
};

extern void __xla_cpu_runtime_ACLMatMulF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, int64_t m, int64_t n, int64_t k,
    int32_t transpose_lhs, int32_t transpose_rhs);

extern void __xla_cpu_runtime_ACLBatchMatMulF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, int64_t m, int64_t n, int64_t k, int64_t batch_size,
    int32_t transpose_lhs, int32_t transpose_rhs);

#else
extern void __xla_cpu_runtime_ACLMatMulF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, int64_t m, int64_t n, int64_t k,
    int32_t transpose_lhs, int32_t transpose_rhs) {
  std::cerr
      << "Attempt to call ACL MatMul runtime library without defining "
         "XLA_CPU_USE_ACL. Add --define=build_with_acl=true to build with ACL.";
  exit(1);
}

extern void __xla_cpu_runtime_ACLBatchMatMulF32(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, float* out,
    float* lhs, float* rhs, int64_t m, int64_t n, int64_t k, int64_t batch_size,
    int32_t transpose_lhs, int32_t transpose_rhs) {
  std::cerr
      << "Attempt to call ACL MatMul runtime library without defining "
         "XLA_CPU_USE_ACL. Add --define=build_with_acl=true to build with ACL.";
  exit(1);
}

#endif  // XLA_CPU_USE_ACL
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_MATMUL_ACL_H_
