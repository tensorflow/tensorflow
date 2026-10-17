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

// XLA lowerings of FlashAttention / FlashAttentionGrad to custom calls (targets
// in flash_attention_custom_call.cc).
#include <cstring>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/kernels/flash_attention/fa2_api.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_custom_call.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op.h"
#include "tensorflow/core/util/determinism.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"

namespace tensorflow {
namespace {

// seed/offset must be uint64_t scalars (they become u64[] custom-call
// operands).
static absl::Status ValidateRngInputs(XlaOpKernelContext* ctx) {
  for (absl::string_view name : {"seed", "offset"}) {
    const TensorShape shape = ctx->InputShape(name);
    if (!TensorShapeUtils::IsScalar(shape)) {
      return errors::InvalidArgument(name, " must be a scalar, got ",
                                     shape.DebugString());
    }
  }
  return absl::OkStatus();
}

static absl::Status ToFlashAttnDtype(DataType dt, int32_t* out) {
  switch (dt) {
    case DT_HALF:
      *out = kFlashAttnF16;
      return absl::OkStatus();
    case DT_BFLOAT16:
      *out = kFlashAttnBF16;
      return absl::OkStatus();
    case DT_FLOAT:
      *out = kFlashAttnF32;
      return absl::OkStatus();
    default:
      return errors::InvalidArgument("unsupported dtype ", DataTypeString(dt));
  }
}

// Same restrictions as the TF kernels in flash_attention_op.cc, except that
// feature sizes over 128 are checked by shape only (no device query at compile
// time): if the device turns out not to support them, the custom call fails at
// run time.
static absl::Status ValidateQKV(const TensorShape& q, const TensorShape& k,
                                const TensorShape& v, bool causal,
                                DataType dt) {
  if (q.dims() != 4 || k.dims() != 4 || v.dims() != 4) {
    return errors::InvalidArgument(
        "Q/K/V must be rank 4 ([batch, seq, heads, dim])");
  }
  if (k.dim_size(0) != q.dim_size(0) || v.dim_size(0) != q.dim_size(0)) {
    return errors::InvalidArgument("Q/K/V batch size mismatch");
  }
  if (k.dim_size(2) != q.dim_size(2) || v.dim_size(2) != q.dim_size(2)) {
    return errors::InvalidArgument("Q/K/V heads mismatch");
  }
  if (k.dim_size(3) != q.dim_size(3)) {
    return errors::InvalidArgument("K feature size mismatch");
  }
  if (v.dim_size(1) != k.dim_size(1)) {
    return errors::InvalidArgument("V seq size mismatch");
  }
  if (causal && q.dim_size(1) != k.dim_size(1)) {
    return errors::InvalidArgument("seq size Q != K while is_causal = True");
  }
  TF_RETURN_IF_ERROR(functor::CheckDimsFitInt32(q.dim_size(0), q.dim_size(2),
                                                q.dim_size(1), k.dim_size(1),
                                                q.dim_size(3), v.dim_size(3)));
  const int64_t d_qk = q.dim_size(3), d_v = v.dim_size(3);
  const bool fa2_type = dt == DT_HALF || dt == DT_BFLOAT16;
  if (std::max(d_qk, d_v) > 128 &&
      !(fa2_type && flash_attention_fa2::ShapeSupported(d_qk, d_v) &&
        flash_attention_fa2::GridSupported(q.dim_size(0), q.dim_size(2)))) {
    return functor::FeatureSizeError(d_qk, d_v);
  }
  return absl::OkStatus();
}

static absl::Status ExpectShape(absl::string_view name, const TensorShape& got,
                                const TensorShape& want) {
  if (got != want) {
    return errors::InvalidArgument(name, " has shape ", got.DebugString(),
                                   ", expected ", want.DebugString());
  }
  return absl::OkStatus();
}

// Dense row-major layout, matching Eigen RowMajor TensorMaps.
static xla::Shape RowMajor(xla::PrimitiveType t,
                           std::initializer_list<int64_t> dims) {
  return xla::ShapeUtil::MakeShapeWithDescendingLayout(t, dims);
}

template <typename Desc>
static std::string ToOpaque(const Desc& d) {
  return std::string(reinterpret_cast<const char*>(&d), sizeof(d));
}

class FlashAttnXlaOp : public XlaOpKernel {
 public:
  explicit FlashAttnXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_causal", &causal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout_rate", &dropout_));
    OP_REQUIRES(ctx, dropout_ >= 0.0f && dropout_ < 1.0f,
                errors::InvalidArgument("dropout_rate must be in [0, 1)"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &scale_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Zeroed, padding included, since the raw bytes become the custom call's
    // opaque string (and so part of the HLO).
    FlashAttnDesc d;
    std::memset(&d, 0, sizeof(d));
    const TensorShape q = ctx->InputShape(0);
    const TensorShape k = ctx->InputShape(1);
    const TensorShape v = ctx->InputShape(2);
    OP_REQUIRES_OK(ctx, ValidateQKV(q, k, v, causal_, ctx->input_type(0)));
    OP_REQUIRES_OK(ctx, ValidateRngInputs(ctx));
    OP_REQUIRES_OK(ctx, ToFlashAttnDtype(ctx->input_type(0), &d.dtype));
    xla::PrimitiveType ptype;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(0), &ptype));

    d.B = q.dim_size(0);
    d.Sq = q.dim_size(1);
    d.H = q.dim_size(2);
    d.Dqk = q.dim_size(3);
    d.Sk = k.dim_size(1);
    d.Dv = v.dim_size(3);
    d.dropout = dropout_;
    d.scale = scale_;
    d.causal = causal_;

    xla::Shape u64_s = RowMajor(xla::U64, {});
    xla::Shape q_s = RowMajor(ptype, {d.B, d.Sq, d.H, d.Dqk});
    xla::Shape k_s = RowMajor(ptype, {d.B, d.Sk, d.H, d.Dqk});
    xla::Shape v_s = RowMajor(ptype, {d.B, d.Sk, d.H, d.Dv});
    xla::Shape out_s = RowMajor(ptype, {d.B, d.Sq, d.H, d.Dv});
    xla::Shape st_s = RowMajor(xla::F32, {d.B, d.H, d.Sq});

    xla::XlaOp call = xla::CustomCallWithLayout(
        ctx->builder(), "tf_flash_attention_fwd",
        /*operands=*/
        {ctx->Input(0), ctx->Input(1), ctx->Input(2), ctx->Input("seed"),
         ctx->Input("offset")},
        /*shape_with_layout=*/xla::ShapeUtil::MakeTupleShape({out_s, st_s}),
        /*operand_shapes_with_layout=*/{q_s, k_s, v_s, u64_s, u64_s},
        /*opaque=*/ToOpaque(d),
        /*has_side_effect=*/false,
        /*output_operand_aliasing=*/{},
        /*literal=*/nullptr,
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/
        xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING);

    ctx->SetOutput(0, xla::GetTupleElement(call, 0));  // Out
    ctx->SetOutput(1, xla::GetTupleElement(call, 1));  // stats
  }

 private:
  bool causal_;
  float dropout_, scale_;
};

class FlashAttnGradXlaOp : public XlaOpKernel {
 public:
  explicit FlashAttnGradXlaOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_causal", &causal_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout_rate", &dropout_));
    OP_REQUIRES(ctx, dropout_ >= 0.0f && dropout_ < 1.0f,
                errors::InvalidArgument("dropout_rate must be in [0, 1)"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &scale_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Zeroed, padding included, since the raw bytes become the custom call's
    // opaque string (and so part of the HLO).
    FlashAttnBwdDesc d;
    std::memset(&d, 0, sizeof(d));
    const TensorShape q = ctx->InputShape(0);
    const TensorShape k = ctx->InputShape(1);
    const TensorShape v = ctx->InputShape(2);
    OP_REQUIRES_OK(ctx, ValidateQKV(q, k, v, causal_, ctx->input_type(0)));
    OP_REQUIRES_OK(ctx, ValidateRngInputs(ctx));
    OP_REQUIRES_OK(ctx, ToFlashAttnDtype(ctx->input_type(0), &d.dtype));
    xla::PrimitiveType ptype;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(0), &ptype));

    d.B = q.dim_size(0);
    d.Sq = q.dim_size(1);
    d.H = q.dim_size(2);
    d.Dqk = q.dim_size(3);
    d.Sk = k.dim_size(1);
    d.Dv = v.dim_size(3);
    d.dropout = dropout_;
    d.scale = scale_;
    d.causal = causal_;
    d.deterministic = OpDeterminismRequired();

    const TensorShape out_shape({d.B, d.Sq, d.H, d.Dv});
    OP_REQUIRES_OK(ctx, ExpectShape("out", ctx->InputShape(3), out_shape));
    OP_REQUIRES_OK(ctx, ExpectShape("stats", ctx->InputShape(4),
                                    TensorShape({d.B, d.H, d.Sq})));
    OP_REQUIRES_OK(ctx, ExpectShape("do", ctx->InputShape(5), out_shape));

    // Sized from shapes alone, so it doesn't matter which device is current
    // here.
    switch (d.dtype) {
      case kFlashAttnF16:
        d.workspace_bytes = functor::FlashAttentionGradFunctor<
            GPUDevice, Eigen::half>::WorkspaceBytes(d.B, d.H, d.Sq, d.Sk, d.Dqk,
                                                    d.Dv, d.deterministic);
        break;
      case kFlashAttnBF16:
        d.workspace_bytes = functor::FlashAttentionGradFunctor<
            GPUDevice, Eigen::bfloat16>::WorkspaceBytes(d.B, d.H, d.Sq, d.Sk,
                                                        d.Dqk, d.Dv,
                                                        d.deterministic);
        break;
      default:
        d.workspace_bytes = functor::FlashAttentionGradFunctor<
            GPUDevice, float>::WorkspaceBytes(d.B, d.H, d.Sq, d.Sk, d.Dqk, d.Dv,
                                              d.deterministic);
    }

    xla::Shape u64_s = RowMajor(xla::U64, {});
    xla::Shape q_s = RowMajor(ptype, {d.B, d.Sq, d.H, d.Dqk});
    xla::Shape k_s = RowMajor(ptype, {d.B, d.Sk, d.H, d.Dqk});
    xla::Shape v_s = RowMajor(ptype, {d.B, d.Sk, d.H, d.Dv});
    xla::Shape o_s = RowMajor(ptype, {d.B, d.Sq, d.H, d.Dv});
    xla::Shape st_s = RowMajor(xla::F32, {d.B, d.H, d.Sq});
    xla::Shape ws_s = RowMajor(xla::U8, {d.workspace_bytes});

    xla::XlaOp call = xla::CustomCallWithLayout(
        ctx->builder(), "tf_flash_attention_bwd",
        /*operands=*/
        {ctx->Input(0), ctx->Input(1), ctx->Input(2), ctx->Input(3),
         ctx->Input(4), ctx->Input(5), ctx->Input("seed"),
         ctx->Input("offset")},
        /*shape_with_layout=*/
        xla::ShapeUtil::MakeTupleShape({q_s, k_s, v_s, ws_s}),
        /*operand_shapes_with_layout=*/
        {q_s, k_s, v_s, o_s, st_s, o_s, u64_s, u64_s},
        /*opaque=*/ToOpaque(d),
        /*has_side_effect=*/false,
        /*output_operand_aliasing=*/{},
        /*literal=*/nullptr,
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/
        xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING);

    ctx->SetOutput(0, xla::GetTupleElement(call, 0));  // dQ
    ctx->SetOutput(1, xla::GetTupleElement(call, 1));  // dK
    ctx->SetOutput(2, xla::GetTupleElement(call, 2));  // dV
  }

 private:
  bool causal_;
  float dropout_, scale_;
};

REGISTER_XLA_OP(Name("FlashAttention")
                    .Device(DEVICE_GPU_XLA_JIT)
                    .TypeConstraint("T", {DT_HALF, DT_BFLOAT16, DT_FLOAT}),
                FlashAttnXlaOp);

REGISTER_XLA_OP(Name("FlashAttentionGrad")
                    .Device(DEVICE_GPU_XLA_JIT)
                    .TypeConstraint("T", {DT_HALF, DT_BFLOAT16, DT_FLOAT}),
                FlashAttnGradXlaOp);

}  // namespace
}  // namespace tensorflow
