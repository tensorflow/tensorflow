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

#include "tensorflow/core/kernels/isotonic_regression_op.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

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
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

struct IsotonicRegressionConfig {
  int64_t batch_size;
  int64_t n;
  DataType input_dtype;
  DataType output_dtype;
};

void XlaIsotonicRegressionCpuCustomCall(void* out, const void** in,
                                        const char* opaque, size_t opaque_len,
                                        XlaCustomCallStatus* status) {
  if (opaque_len < sizeof(IsotonicRegressionConfig)) {
    const char* msg =
        "Invalid config payload for IsotonicRegression CustomCall";
    XlaCustomCallStatusSetFailure(status, msg, strlen(msg));
    return;
  }
  IsotonicRegressionConfig config;
  std::memcpy(&config, opaque, sizeof(config));

  if (config.batch_size <= 0 || config.n <= 0) {
    XlaCustomCallStatusSetSuccess(status);
    return;
  }

  void** out_ptrs = reinterpret_cast<void**>(out);
  void* solution_buf = out_ptrs[0];
  int32_t* segments_buf = reinterpret_cast<int32_t*>(out_ptrs[1]);
  const void* input_buf = in[0];

#define DISPATCH_PAVA(Tin, Tout)                                \
  isotonic_regression::SolvePavaBatch<Tin, Tout>(               \
      reinterpret_cast<const Tin*>(input_buf),                  \
      reinterpret_cast<Tout*>(solution_buf), segments_buf,      \
      config.batch_size, config.n);

  if (config.input_dtype == DT_FLOAT && config.output_dtype == DT_FLOAT) {
    DISPATCH_PAVA(float, float);
  } else if (config.input_dtype == DT_DOUBLE &&
             config.output_dtype == DT_DOUBLE) {
    DISPATCH_PAVA(double, double);
  } else if (config.input_dtype == DT_BFLOAT16 &&
             config.output_dtype == DT_BFLOAT16) {
    DISPATCH_PAVA(bfloat16, bfloat16);
  } else if (config.input_dtype == DT_HALF && config.output_dtype == DT_HALF) {
    DISPATCH_PAVA(Eigen::half, Eigen::half);
  } else if (config.input_dtype == DT_INT32 &&
             config.output_dtype == DT_DOUBLE) {
    DISPATCH_PAVA(int32, double);
  } else if (config.input_dtype == DT_INT64 &&
             config.output_dtype == DT_DOUBLE) {
    DISPATCH_PAVA(int64_t, double);
  } else if (config.input_dtype == DT_INT16 &&
             config.output_dtype == DT_FLOAT) {
    DISPATCH_PAVA(int16, float);
  } else if (config.input_dtype == DT_INT8 && config.output_dtype == DT_FLOAT) {
    DISPATCH_PAVA(int8, float);
  } else {
    DISPATCH_PAVA(float, float);
  }
#undef DISPATCH_PAVA

  XlaCustomCallStatusSetSuccess(status);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "XlaIsotonicRegressionCpuCustomCall", XlaIsotonicRegressionCpuCustomCall);

class IsotonicRegressionOp : public XlaOpKernel {
 public:
  explicit IsotonicRegressionOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dtype", &output_dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const DataType input_dtype = ctx->input_type(0);

    OP_REQUIRES(ctx, input_shape.dims() >= 1,
                errors::InvalidArgument("Input must have rank >= 1, got shape ",
                                        input_shape.DebugString()));

    int64_t n = input_shape.dim_size(input_shape.dims() - 1);
    int64_t batch_size = input_shape.num_elements() / (n > 0 ? n : 1);

    xla::Shape input_xla_shape;
    OP_REQUIRES_OK(
        ctx, TensorShapeToXLAShape(input_dtype, input_shape, &input_xla_shape));

    xla::Shape output_xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(output_dtype_, input_shape,
                                              &output_xla_shape));

    xla::Shape segments_xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_INT32, input_shape,
                                              &segments_xla_shape));

    xla::Shape tuple_shape =
        xla::ShapeUtil::MakeTupleShape({output_xla_shape, segments_xla_shape});

    IsotonicRegressionConfig config;
    config.batch_size = batch_size;
    config.n = n;
    config.input_dtype = input_dtype;
    config.output_dtype = output_dtype_;

    std::string opaque(reinterpret_cast<const char*>(&config), sizeof(config));

    xla::XlaOp result = xla::CustomCall(
        ctx->builder(), "XlaIsotonicRegressionCpuCustomCall", {ctx->Input(0)},
        tuple_shape, opaque,
        /*has_side_effect=*/false, /*output_operand_aliasing=*/{},
        /*literal=*/nullptr, xla::CustomCallSchedule::SCHEDULE_NONE,
        xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED);

    ctx->SetOutput(0, xla::GetTupleElement(result, 0));
    ctx->SetOutput(1, xla::GetTupleElement(result, 1));
  }

 private:
  DataType output_dtype_;
};

REGISTER_XLA_OP(Name("IsotonicRegression").Device(DEVICE_CPU_XLA_JIT),
                IsotonicRegressionOp);

}  // namespace
}  // namespace tensorflow

