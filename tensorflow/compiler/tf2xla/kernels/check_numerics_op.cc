/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/sharding_builder.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace {

class CheckNumericsOp : public XlaOpKernel {
 public:
  explicit CheckNumericsOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("message", &message_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp input = ctx->Input(0);

    bool is_gpu = false;
    if (ctx->compiler() != nullptr) {
      absl::string_view device_type =
          ctx->compiler()->options().device_type.type_string();
      if (device_type == "GPU" || device_type == "XLA_GPU" ||
          device_type == "XLA_GPU_JIT") {
        is_gpu = true;
      }
    }

    if (is_gpu) {
      xla::XlaOp is_nan = xla::IsNan(input);
      xla::XlaOp is_inf = xla::IsInf(input);
      xla::XlaOp is_nan_or_inf = xla::Or(is_nan, is_inf);

      xla::XlaOp has_nan_or_inf =
          xla::ReduceAll(is_nan_or_inf, xla::ConstantR0<bool>(b, false),
                         xla::CreateScalarOrComputation(xla::PRED, b));

      xla::XlaOp is_ok = xla::Not(has_nan_or_inf);

      std::string escaped_message;
      for (char c : message_) {
        if (c == '"') {
          escaped_message += "\\\"";
        } else if (c == '\\') {
          escaped_message += "\\\\";
        } else if (c == '\n') {
          escaped_message += "\\n";
        } else if (c == '\r') {
          escaped_message += "\\r";
        } else if (c == '\t') {
          escaped_message += "\\t";
        } else {
          escaped_message += c;
        }
      }

      std::string backend_config = absl::StrFormat(
          "{error_msg = \"CheckNumerics failed: %s\"}", escaped_message);

      xla::XlaOp check = xla::CustomCall(
          b, "__xla_gpu_assert", {is_ok}, xla::ShapeUtil::MakeTokenShape(),
          backend_config,
          /*has_side_effect=*/true,
          /*output_operand_aliasing=*/{},
          /*literal=*/nullptr, xla::CustomCallSchedule::SCHEDULE_NONE,
          xla::CustomCallApiVersion::API_VERSION_TYPED_FFI);

      auto sharding_or = b->GetOpSharding(input);
      if (sharding_or.ok() && sharding_or.value().has_value()) {
        OP_REQUIRES_OK(ctx,
                       b->SetInstructionSharding(check, sharding_or.value()));
      } else {
        OP_REQUIRES_OK(ctx, b->SetInstructionSharding(
                                check, xla::sharding_builder::Replicate()));
      }
    } else {
      static mutex mu(tensorflow::LINKER_INITIALIZED);
      static int log_counter = 0;
      mutex_lock l(mu);
      if (log_counter < 20) {
        ++log_counter;
        LOG(WARNING) << "Ignoring CheckNumerics operator " << name()
                     << " on non-GPU backend";
      }
    }

    ctx->SetOutput(0, input);
  }

 private:
  std::string message_;
};

REGISTER_XLA_OP(Name("CheckNumerics"), CheckNumericsOp);

}  // anonymous namespace
}  // namespace tensorflow
