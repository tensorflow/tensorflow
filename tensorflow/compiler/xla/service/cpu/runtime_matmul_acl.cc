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

#ifdef XLA_CPU_USE_ACL
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul_acl.h"

#include "absl/base/call_once.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_lightweight_check.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/types.h"

#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/tsl/platform/dynamic_annotations.h"

namespace {
// ACL GEMM API for 32-bit Matrix Multiplication.

// MatMul function is defined as: c = alpha * op(a) * op(b) + beta * c.
// Since XLA MatMul does not use alpha, beta, we set them to 1.0 and 0.0.
// Matrix lhs, rhs and out are all column-major.
int32_t MatMulF32(const void* run_options_ptr, float* out, float* lhs,
                  float* rhs, int64_t m, int64_t n, int64_t k,
                  int64_t batch_size, int32_t transpose_lhs,
                  int32_t transpose_rhs) {
  const float alpha = 1.0f, beta = 0.0f;

  /* TODO: optimize this object creation along with tensor init and
   * gemm configuration by caching the shapes, similar to onednn
   * primitive caching feature
   */
  struct acl_matmul_obj_t acl_obj;
  struct acl_matmul_conf_t acl_conf;

  acl_conf.is_trans_lhs = (bool)transpose_lhs;
  acl_conf.is_trans_rhs = (bool)transpose_rhs;

  if (acl_conf.is_trans_lhs) {
    acl_conf.lhs_acc_info =
        arm_compute::TensorInfo(arm_compute::TensorShape(k, m, batch_size), 1,
                                arm_compute::DataType::F32);
  }
  if (acl_conf.is_trans_rhs) {
    acl_conf.rhs_acc_info =
        arm_compute::TensorInfo(arm_compute::TensorShape(n, k, 1, batch_size),
                                1, arm_compute::DataType::F32);
  }

  acl_conf.lhs_info =
      arm_compute::TensorInfo(arm_compute::TensorShape(m, k, batch_size), 1,
                              arm_compute::DataType::F32);
  acl_conf.rhs_info =
      arm_compute::TensorInfo(arm_compute::TensorShape(k, n, 1, batch_size), 1,
                              arm_compute::DataType::F32);
  acl_conf.out_info =
      arm_compute::TensorInfo(arm_compute::TensorShape(m, n, 1, batch_size), 1,
                              arm_compute::DataType::F32);

  /* TODO: add TF_XLA_* flag for runtime control of fast math mode*/
  bool is_fastmath_enabled = true;
  acl_conf.gemm_info.set_fast_math(is_fastmath_enabled);

  // Fused ReLU activation
  acl_conf.gemm_info.set_activation_info(arm_compute::ActivationLayerInfo());

  // Set alpha (output scaling)
  acl_conf.alpha = alpha;

  // Validate ACL transpose
  if (acl_conf.is_trans_lhs) {
    auto acl_trans_lhs_st = arm_compute::NETranspose::validate(
        &acl_conf.lhs_acc_info, &acl_conf.lhs_info);
    if (acl_trans_lhs_st.error_code() != arm_compute::ErrorCode::OK) {
      VLOG(1) << "lhs transpose validation failed";
      return -1;
    }
  }
  if (acl_conf.is_trans_rhs) {
    auto acl_trans_rhs_st = arm_compute::NETranspose::validate(
        &acl_conf.rhs_acc_info, &acl_conf.rhs_info);
    if (acl_trans_rhs_st.error_code() != arm_compute::ErrorCode::OK) {
      VLOG(1) << "rhs transpose validation failed";
      return -1;
    }
  }

  // Validate ACL GEMM
  auto acl_st = arm_compute::NEGEMM::validate(
      &acl_conf.rhs_info, &acl_conf.lhs_info, nullptr, &acl_conf.out_info,
      acl_conf.alpha, 0.0f, acl_conf.gemm_info);
  if (acl_st.error_code() != arm_compute::ErrorCode::OK) {
    VLOG(1) << "validate acl GEMM FAILED";
    return -1;
  }

  static absl::once_flag flag_once;
  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(run_options_ptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  const Eigen::ThreadPoolDevice* tpd =
      (Eigen::ThreadPoolDevice*)(run_options->intra_op_thread_pool());
  // The threads in Compute Library are bound for the cores 0..max_threads-1
  const int max_threads = tpd->numThreads();

  // arm_compute::Scheduler does not support concurrent access thus a
  // workaround here restricts it to only one call
  absl::call_once(flag_once, [&]() {
    arm_compute::Scheduler::get().set_num_threads(max_threads);
  });

  // configure the acl obj with the config
  acl_obj.lhs_tensor.allocator()->init(acl_conf.lhs_info);
  acl_obj.rhs_tensor.allocator()->init(acl_conf.rhs_info);
  acl_obj.out_tensor.allocator()->init(acl_conf.out_info);

  // Configure transpose kernel for src, wei or both
  if (acl_conf.is_trans_lhs) {
    acl_obj.lhs_acc_tensor.allocator()->init(acl_conf.lhs_acc_info);
    acl_obj.trans_lhs.configure(&acl_obj.lhs_acc_tensor, &acl_obj.lhs_tensor);
  }
  if (acl_conf.is_trans_rhs) {
    acl_obj.rhs_acc_tensor.allocator()->init(acl_conf.rhs_acc_info);
    acl_obj.trans_rhs.configure(&acl_obj.rhs_acc_tensor, &acl_obj.rhs_tensor);
  }
  // Configure GEMM
  acl_obj.gemm.configure(&acl_obj.rhs_tensor, &acl_obj.lhs_tensor, nullptr,
                         &acl_obj.out_tensor, acl_conf.alpha, 0.0f,
                         acl_conf.gemm_info);

  // Run transpose kernel
  if (transpose_lhs && !transpose_rhs) {
    acl_obj.lhs_tensor.allocator()->allocate();
    acl_obj.lhs_acc_tensor.allocator()->import_memory(lhs);
    acl_obj.trans_lhs.run();
    acl_obj.rhs_tensor.allocator()->import_memory(rhs);
  } else if (transpose_rhs && !transpose_lhs) {
    acl_obj.rhs_tensor.allocator()->allocate();
    acl_obj.rhs_acc_tensor.allocator()->import_memory(rhs);
    acl_obj.trans_rhs.run();
    acl_obj.lhs_tensor.allocator()->import_memory(lhs);
  } else if (transpose_rhs && transpose_lhs) {
    acl_obj.lhs_tensor.allocator()->allocate();
    acl_obj.lhs_acc_tensor.allocator()->import_memory(lhs);
    acl_obj.rhs_tensor.allocator()->allocate();
    acl_obj.rhs_acc_tensor.allocator()->import_memory(rhs);
    acl_obj.trans_lhs.run();
    acl_obj.trans_rhs.run();
  } else {
    acl_obj.lhs_tensor.allocator()->import_memory(lhs);
    acl_obj.rhs_tensor.allocator()->import_memory(rhs);
  }

  acl_obj.out_tensor.allocator()->import_memory(out);

  // Execute the function
  acl_obj.gemm.run();

  acl_obj.lhs_tensor.allocator()->free();
  acl_obj.rhs_tensor.allocator()->free();
  acl_obj.out_tensor.allocator()->free();
  if (acl_conf.is_trans_lhs) acl_obj.lhs_acc_tensor.allocator()->free();
  if (acl_conf.is_trans_rhs) acl_obj.rhs_acc_tensor.allocator()->free();

  return 0;
}

}  // namespace

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_ACLMatMulF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs, int64_t m,
    int64_t n, int64_t k, int32_t transpose_lhs, int32_t transpose_rhs) {
  if (MatMulF32(run_options_ptr, out, lhs, rhs, m, n, k, 1 /*batch_size*/,
                transpose_lhs, transpose_rhs) < 0) {
    VLOG(1) << "ACL matmul failed, fallback to Eigen matmul";
    __xla_cpu_runtime_EigenMatMulF32(run_options_ptr, out, lhs, rhs, m, n, k,
                                     transpose_lhs, transpose_rhs);
  }
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_ACLBatchMatMulF32(
    const void* run_options_ptr, float* out, float* lhs, float* rhs, int64_t m,
    int64_t n, int64_t k, int64_t batch_size, int32_t transpose_lhs,
    int32_t transpose_rhs) {
  if (MatMulF32(run_options_ptr, out, lhs, rhs, m, n, k, batch_size,
                transpose_lhs, transpose_rhs) < 0) {
    VLOG(1) << "ACL batch matmul failed, fallback to Eigen batch matmul";
    __xla_cpu_runtime_EigenBatchMatMulF32(run_options_ptr, out, lhs, rhs, m, n,
                                          k, batch_size, transpose_lhs,
                                          transpose_rhs);
  }
}

#endif  // XLA_CPU_USE_ACL
