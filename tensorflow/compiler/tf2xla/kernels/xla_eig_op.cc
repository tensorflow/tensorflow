/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

#include "Eigen/Core"  // from @eigen_archive
#include "Eigen/Eigenvalues"  // from @eigen_archive
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

struct EigConfig {
  int64_t batch_size;
  int64_t n;
  DataType input_dtype;
  DataType output_dtype;
  bool compute_v;
};

template <typename InputScalar, typename OutputScalar>
absl::Status ComputeEigBatch(const InputScalar* input,
                             OutputScalar* eigenvalues,
                             OutputScalar* eigenvectors,
                             int64_t batch_size,
                             int64_t n,
                             bool compute_v) {
  if (batch_size <= 0 || n <= 0) {
    return absl::OkStatus();
  }

  // This algorithm relies on denormals, so switch them back on locally.
  port::ScopedDontFlushDenormal dont_flush_denormals;

  using InputMatrix = Eigen::Matrix<InputScalar, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>;
  using InputConstMatrixMap = Eigen::Map<const InputMatrix>;
  using OutputMatrix = Eigen::Matrix<OutputScalar, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>;
  using OutputMatrixMap = Eigen::Map<OutputMatrix>;
  using OutputVectorMap =
      Eigen::Map<Eigen::Matrix<OutputScalar, 1, Eigen::Dynamic>>;

  using EigenSolver =
      std::conditional_t<Eigen::NumTraits<InputScalar>::IsComplex,
                         Eigen::ComplexEigenSolver<InputMatrix>,
                         Eigen::EigenSolver<InputMatrix>>;

  // Reused across the entire batch to eliminate O(batch_size) dynamic
  // allocations.
  EigenSolver solver;
  const int64_t matrix_size = n * n;

  for (int64_t b = 0; b < batch_size; ++b) {
    InputConstMatrixMap input_map(input + b * matrix_size, n, n);
    solver.compute(input_map, compute_v);
    if (solver.info() != Eigen::Success) {
      return absl::InvalidArgumentError(
          "Eigen decomposition was not successful. The input might not be "
          "valid.");
    }

    OutputVectorMap e_map(eigenvalues + b * n, 1, n);
    e_map = solver.eigenvalues().template cast<OutputScalar>();

    if (compute_v) {
      OutputMatrixMap v_map(eigenvectors + b * matrix_size, n, n);
      v_map = solver.eigenvectors();
    }
  }

  return absl::OkStatus();
}

void XlaEigCpuCustomCall(void* out, const void** in, const char* opaque,
                         size_t opaque_len, XlaCustomCallStatus* status) {
  if (opaque_len != sizeof(EigConfig)) {
    const char* msg = "Invalid config payload size for Eig CustomCall";
    XlaCustomCallStatusSetFailure(status, msg, strlen(msg));
    return;
  }
  EigConfig config;
  std::memcpy(&config, opaque, sizeof(config));

  if (config.batch_size <= 0 || config.n <= 0) {
    XlaCustomCallStatusSetSuccess(status);
    return;
  }

  void** out_ptrs = reinterpret_cast<void**>(out);
  void* e_buf = out_ptrs[0];
  void* v_buf = out_ptrs[1];
  const void* input_buf = in[0];

  absl::Status compute_status;
  if (config.input_dtype == DT_FLOAT && config.output_dtype == DT_COMPLEX64) {
    compute_status = ComputeEigBatch<float, complex64>(
        reinterpret_cast<const float*>(input_buf),
        reinterpret_cast<complex64*>(e_buf),
        reinterpret_cast<complex64*>(v_buf),
        config.batch_size, config.n, config.compute_v);
  } else if (config.input_dtype == DT_DOUBLE &&
             config.output_dtype == DT_COMPLEX128) {
    compute_status = ComputeEigBatch<double, complex128>(
        reinterpret_cast<const double*>(input_buf),
        reinterpret_cast<complex128*>(e_buf),
        reinterpret_cast<complex128*>(v_buf),
        config.batch_size, config.n, config.compute_v);
  } else if (config.input_dtype == DT_COMPLEX64 &&
             config.output_dtype == DT_COMPLEX64) {
    compute_status = ComputeEigBatch<complex64, complex64>(
        reinterpret_cast<const complex64*>(input_buf),
        reinterpret_cast<complex64*>(e_buf),
        reinterpret_cast<complex64*>(v_buf),
        config.batch_size, config.n, config.compute_v);
  } else if (config.input_dtype == DT_COMPLEX128 &&
             config.output_dtype == DT_COMPLEX128) {
    compute_status = ComputeEigBatch<complex128, complex128>(
        reinterpret_cast<const complex128*>(input_buf),
        reinterpret_cast<complex128*>(e_buf),
        reinterpret_cast<complex128*>(v_buf),
        config.batch_size, config.n, config.compute_v);
  } else {
    compute_status = absl::InvalidArgumentError(
        absl::StrCat("Unsupported dtype combination in Eig CustomCall: input=",
                     DataTypeString(config.input_dtype),
                     ", output=", DataTypeString(config.output_dtype)));
  }

  if (!compute_status.ok()) {
    std::string msg = std::string(compute_status.message());
    XlaCustomCallStatusSetFailure(status, msg.c_str(), msg.length());
    return;
  }

  XlaCustomCallStatusSetSuccess(status);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("XlaEigCpuCustomCall",
                                             XlaEigCpuCustomCall);

class XlaEigOp : public XlaOpKernel {
 public:
  explicit XlaEigOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("compute_v", &compute_v_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const DataType input_dtype = ctx->input_type(0);

    OP_REQUIRES(ctx, input_shape.dims() >= 2,
                errors::InvalidArgument("Input must have rank >= 2, got shape ",
                                        input_shape.DebugString()));

    int64_t n = input_shape.dim_size(input_shape.dims() - 1);
    OP_REQUIRES(
        ctx, input_shape.dim_size(input_shape.dims() - 2) == n,
        errors::InvalidArgument(
            "Innermost 2 dimensions of input must be square, got shape ",
            input_shape.DebugString()));

    int64_t batch_size = 1;
    for (int i = 0; i < input_shape.dims() - 2; ++i) {
      batch_size *= input_shape.dim_size(i);
    }

    TensorShape e_shape = input_shape;
    e_shape.RemoveLastDims(1);

    xla::Shape e_xla_shape;
    OP_REQUIRES_OK(
        ctx, TensorShapeToXLAShape(output_dtype_, e_shape, &e_xla_shape));

    xla::Shape v_xla_shape;
    if (compute_v_) {
      OP_REQUIRES_OK(
          ctx, TensorShapeToXLAShape(output_dtype_, input_shape, &v_xla_shape));
    } else {
      TensorShape v_shape({0});
      OP_REQUIRES_OK(
          ctx, TensorShapeToXLAShape(output_dtype_, v_shape, &v_xla_shape));
    }

    xla::Shape tuple_shape =
        xla::ShapeUtil::MakeTupleShape({e_xla_shape, v_xla_shape});

    EigConfig config;
    std::memset(&config, 0, sizeof(config));
    config.batch_size = batch_size;
    config.n = n;
    config.input_dtype = input_dtype;
    config.output_dtype = output_dtype_;
    config.compute_v = compute_v_;

    std::string opaque(reinterpret_cast<const char*>(&config), sizeof(config));

    xla::XlaOp result = xla::CustomCall(
        ctx->builder(), "XlaEigCpuCustomCall", {ctx->Input(0)},
        tuple_shape, opaque,
        /*has_side_effect=*/false, /*output_operand_aliasing=*/{},
        /*literal=*/nullptr, xla::CustomCallSchedule::SCHEDULE_NONE,
        xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED);

    ctx->SetOutput(0, xla::GetTupleElement(result, 0));
    ctx->SetOutput(1, xla::GetTupleElement(result, 1));
  }

 private:
  bool compute_v_;
  DataType output_dtype_;
};

REGISTER_XLA_OP(Name("Eig").Device(DEVICE_CPU_XLA_JIT), XlaEigOp);

}  // namespace
}  // namespace tensorflow
