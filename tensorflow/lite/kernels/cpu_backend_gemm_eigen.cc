/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TFLITE_WITH_RUY

#include "tensorflow/lite/kernels/cpu_backend_gemm_eigen.h"

// See b/131835803: in TFLite code, because eigen_spatial_convolutions.h does
// #define Eigen EigenForTFLite, it is difficult to have any #include of Eigen
// headers in a header file, as that results in name classes (compilation
// errors) depending on the order in which these headers are #included.
// So we have moved the #include of Eigen here, in a .cc file, where we have
// control over the header #include sequence.
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

// This function is out-of-line in a .cc file because of the issue
// noted above in the comment on the #include for Eigen/Core.
void GemmImplUsingEigen::Run(
    const MatrixParams<float>& lhs_params, const float* lhs_data,
    const MatrixParams<float>& rhs_params, const float* rhs_data,
    const MatrixParams<float>& dst_params, float* dst_data,
    const GemmParams<float, float>& params, CpuBackendContext* /* context */) {
  // This code assumes specific storage orders, encoded in these Eigen types.
  // These assumptions have been checked by TF_LITE_ASSERT's in the public
  // Gemm entry point already, before the implementation gets to this point.
  using EigenMatrixMapRowMajorConst =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>;
  using EigenMatrixMapColMajorConst =
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::ColMajor>>;
  using EigenMatrixMapColMajorMutable = Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

  EigenMatrixMapRowMajorConst eigen_lhs(lhs_data, lhs_params.rows,
                                        lhs_params.cols);
  EigenMatrixMapColMajorConst eigen_rhs(rhs_data, rhs_params.rows,
                                        rhs_params.cols);
  EigenMatrixMapColMajorMutable eigen_dst(dst_data, dst_params.rows,
                                          dst_params.cols);

  if (rhs_params.cols == 1) {
    eigen_dst.col(0).noalias() = eigen_lhs * eigen_rhs.col(0);
  } else if (lhs_params.rows == 1) {
    eigen_dst.row(0).noalias() = eigen_lhs.row(0) * eigen_rhs;
  } else {
    eigen_dst.noalias() = eigen_lhs * eigen_rhs;
  }

  if (params.bias) {
    BiasAndClamp(params.clamp_min, params.clamp_max, dst_params.rows,
                 params.bias, dst_params.rows * dst_params.cols, dst_data);
  } else {
    eigen_dst = eigen_dst.cwiseMin(params.clamp_max).cwiseMax(params.clamp_min);
  }
}

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // not TFLITE_WITH_RUY
