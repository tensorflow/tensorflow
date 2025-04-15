/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/cast_op.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/cast_op_impl.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define CURRY_TYPES2(FN, arg0)   \
  FN(arg0, bool);                \
  FN(arg0, uint8);               \
  FN(arg0, uint16);              \
  FN(arg0, uint32);              \
  FN(arg0, uint64);              \
  FN(arg0, int8);                \
  FN(arg0, int16);               \
  FN(arg0, int32);               \
  FN(arg0, int64_t);             \
  FN(arg0, Eigen::half);         \
  FN(arg0, bfloat16);            \
  FN(arg0, float);               \
  FN(arg0, double);              \
  FN(arg0, std::complex<float>); \
  FN(arg0, std::complex<double>)

CastOpBase::CastOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &external_src_dtype_));

  OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &external_dst_dtype_));

  OP_REQUIRES_OK(ctx, ctx->GetAttr("Truncate", &use_truncation_));

  // Quantized data types use the same underlying format as their non quantized
  // version so we use the non quantized implementation for casting.
  if (external_dst_dtype_ == DT_QUINT8) {
    dst_dtype_ = DT_UINT8;
  } else if (external_dst_dtype_ == DT_QINT8) {
    dst_dtype_ = DT_INT8;
  } else if (external_dst_dtype_ == DT_QINT32) {
    dst_dtype_ = DT_INT32;
  } else if (external_dst_dtype_ == DT_QINT16) {
    dst_dtype_ = DT_INT16;
  } else if (external_dst_dtype_ == DT_QUINT16) {
    dst_dtype_ = DT_UINT16;
  } else {
    dst_dtype_ = external_dst_dtype_;
  }

  if (external_src_dtype_ == DT_QUINT8) {
    src_dtype_ = DT_UINT8;
  } else if (external_src_dtype_ == DT_QINT8) {
    src_dtype_ = DT_INT8;
  } else if (external_src_dtype_ == DT_QINT32) {
    src_dtype_ = DT_INT32;
  } else if (external_src_dtype_ == DT_QINT16) {
    src_dtype_ = DT_INT16;
  } else if (external_src_dtype_ == DT_QUINT16) {
    src_dtype_ = DT_UINT16;
  } else {
    src_dtype_ = external_src_dtype_;
  }
}

void CastOpBase::Compute(OpKernelContext* ctx) {
  const Tensor& inp = ctx->input(0);
  if (work_ == nullptr) {
    ctx->set_output(0, inp);
  } else if (external_src_dtype_ != src_dtype_ ||
             external_dst_dtype_ != dst_dtype_) {
    Tensor in;
    // If the type is a quantized type we need to do a bitcast since the
    // src_dtype_ is different from external_src_type_.
    OP_REQUIRES_OK(ctx, in.BitcastFrom(inp, src_dtype_, inp.shape()));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &out));
    out->set_dtype(dst_dtype_);
    work_(ctx, in, out, use_truncation_);
    out->set_dtype(external_dst_dtype_);
  } else {
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &out));
    work_(ctx, inp, out, use_truncation_);
  }
}

absl::Status CastOpBase::Unimplemented() {
  return errors::Unimplemented("Cast ", DataTypeString(external_src_dtype_),
                               " to ", DataTypeString(external_dst_dtype_),
                               " is not supported");
}

CpuCastOp::CpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
  OP_REQUIRES_OK(ctx, Prepare());
}

absl::Status CpuCastOp::Prepare() {
  if (external_src_dtype_ == external_dst_dtype_) {
    work_ = nullptr;  // Identity
    return absl::OkStatus();
  }
  if (src_dtype_ == DT_BOOL) {
    work_ = GetCpuCastFromBool(dst_dtype_);
  } else if (src_dtype_ == DT_UINT8) {
    work_ = GetCpuCastFromUint8(dst_dtype_);
  } else if (src_dtype_ == DT_UINT16) {
    work_ = GetCpuCastFromUint16(dst_dtype_);
  } else if (src_dtype_ == DT_UINT32) {
    work_ = GetCpuCastFromUint32(dst_dtype_);
  } else if (src_dtype_ == DT_UINT64) {
    work_ = GetCpuCastFromUint64(dst_dtype_);
  } else if (src_dtype_ == DT_INT8) {
    work_ = GetCpuCastFromInt8(dst_dtype_);
  } else if (src_dtype_ == DT_INT16) {
    work_ = GetCpuCastFromInt16(dst_dtype_);
  } else if (src_dtype_ == DT_INT32) {
    work_ = GetCpuCastFromInt32(dst_dtype_);
  } else if (src_dtype_ == DT_INT64) {
    work_ = GetCpuCastFromInt64(dst_dtype_);
  } else if (src_dtype_ == DT_HALF) {
    work_ = GetCpuCastFromHalf(dst_dtype_);
  } else if (src_dtype_ == DT_FLOAT) {
    work_ = GetCpuCastFromFloat(dst_dtype_);
  } else if (src_dtype_ == DT_DOUBLE) {
    work_ = GetCpuCastFromDouble(dst_dtype_);
  } else if (src_dtype_ == DT_COMPLEX64) {
    work_ = GetCpuCastFromComplex64(dst_dtype_);
  } else if (src_dtype_ == DT_COMPLEX128) {
    work_ = GetCpuCastFromComplex128(dst_dtype_);
  } else if (src_dtype_ == DT_BFLOAT16) {
    work_ = GetCpuCastFromBfloat(dst_dtype_);
  } else if (src_dtype_ == DT_FLOAT8_E5M2) {
    work_ = GetCpuCastFromFloat8e5m2(dst_dtype_);
  } else if (src_dtype_ == DT_FLOAT8_E4M3FN) {
    work_ = GetCpuCastFromFloat8e4m3fn(dst_dtype_);
  } else if (src_dtype_ == DT_INT4) {
    work_ = GetCpuCastFromInt4(dst_dtype_);
  } else if (src_dtype_ == DT_UINT4) {
    work_ = GetCpuCastFromUint4(dst_dtype_);
  }

  // TODO(sesse): If CPU casting to or from Eigen::half ever becomes a
  // bottleneck, we could probably implement specialized support for
  // vectorized versions (not the least based on F16C for Haswell
  // or newer).

  return work_ == nullptr ? Unimplemented() : absl::OkStatus();
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
class GpuCastOp : public CastOpBase {
 public:
  explicit GpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 private:
  Status Prepare() {
    if (external_src_dtype_ == external_dst_dtype_) {
      work_ = nullptr;  // Identity
      return OkStatus();
    }
    if (src_dtype_ == DT_BOOL) {
      work_ = GetGpuCastFromBool(dst_dtype_);
    } else if (src_dtype_ == DT_UINT8) {
      work_ = GetGpuCastFromUint8(dst_dtype_);
    } else if (src_dtype_ == DT_UINT16) {
      work_ = GetGpuCastFromUint16(dst_dtype_);
    } else if (src_dtype_ == DT_UINT32) {
      work_ = GetGpuCastFromUint32(dst_dtype_);
    } else if (src_dtype_ == DT_UINT64) {
      work_ = GetGpuCastFromUint64(dst_dtype_);
    } else if (src_dtype_ == DT_INT8) {
      work_ = GetGpuCastFromInt8(dst_dtype_);
    } else if (src_dtype_ == DT_INT16) {
      work_ = GetGpuCastFromInt16(dst_dtype_);
    } else if (src_dtype_ == DT_INT32) {
      work_ = GetGpuCastFromInt32(dst_dtype_);
    } else if (src_dtype_ == DT_INT64) {
      work_ = GetGpuCastFromInt64(dst_dtype_);
    } else if (src_dtype_ == DT_HALF) {
      work_ = GetGpuCastFromHalf(dst_dtype_);
    } else if (src_dtype_ == DT_FLOAT) {
      work_ = GetGpuCastFromFloat(dst_dtype_);
    } else if (src_dtype_ == DT_DOUBLE) {
      work_ = GetGpuCastFromDouble(dst_dtype_);
    } else if (src_dtype_ == DT_COMPLEX64) {
      work_ = GetGpuCastFromComplex64(dst_dtype_);
    } else if (src_dtype_ == DT_COMPLEX128) {
      work_ = GetGpuCastFromComplex128(dst_dtype_);
    } else if (src_dtype_ == DT_BFLOAT16) {
      work_ = GetGpuCastFromBfloat(dst_dtype_);
    } else if (src_dtype_ == DT_FLOAT8_E5M2) {
      work_ = GetGpuCastFromFloat8e5m2(dst_dtype_);
    } else if (src_dtype_ == DT_FLOAT8_E4M3FN) {
      work_ = GetGpuCastFromFloat8e4m3fn(dst_dtype_);
    } else if (src_dtype_ == DT_INT4) {
      work_ = GetGpuCastFromInt4(dst_dtype_);
    } else if (src_dtype_ == DT_UINT4) {
      work_ = GetGpuCastFromUint4(dst_dtype_);
    }
    return work_ == nullptr ? Unimplemented() : OkStatus();
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef CAST_CASE

REGISTER_KERNEL_BUILDER(Name("Cast").Device(DEVICE_CPU), CpuCastOp);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define REGISTER_CAST_GPU(srctype, dsttype)                    \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_GPU),             \
                          GpuCastOp)

#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
CURRY_TYPES2(REGISTER_CAST_GPU, bool);
CURRY_TYPES2(REGISTER_CAST_GPU, int8);
CURRY_TYPES2(REGISTER_CAST_GPU, int16);
CURRY_TYPES2(REGISTER_CAST_GPU, int32);
CURRY_TYPES2(REGISTER_CAST_GPU, int64);
CURRY_TYPES2(REGISTER_CAST_GPU, uint8);
CURRY_TYPES2(REGISTER_CAST_GPU, uint16);
CURRY_TYPES2(REGISTER_CAST_GPU, uint32);
CURRY_TYPES2(REGISTER_CAST_GPU, uint64);
CURRY_TYPES2(REGISTER_CAST_GPU, Eigen::half);
CURRY_TYPES2(REGISTER_CAST_GPU, float);
CURRY_TYPES2(REGISTER_CAST_GPU, double);
CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<float>);
CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<double>);
#else
REGISTER_CAST_GPU(bool, bfloat16);
REGISTER_CAST_GPU(int8, bfloat16);
REGISTER_CAST_GPU(int16, bfloat16);
REGISTER_CAST_GPU(int32, bfloat16);
REGISTER_CAST_GPU(int64, bfloat16);
REGISTER_CAST_GPU(uint8, bfloat16);
REGISTER_CAST_GPU(uint16, bfloat16);
REGISTER_CAST_GPU(uint32, bfloat16);
REGISTER_CAST_GPU(uint64, bfloat16);
REGISTER_CAST_GPU(Eigen::half, bfloat16);
REGISTER_CAST_GPU(float, bfloat16);
REGISTER_CAST_GPU(double, bfloat16);
REGISTER_CAST_GPU(std::complex<float>, bfloat16);
REGISTER_CAST_GPU(std::complex<double>, bfloat16);
#endif
CURRY_TYPES2(REGISTER_CAST_GPU, bfloat16);

REGISTER_CAST_GPU(float, float8_e5m2);
REGISTER_CAST_GPU(float, float8_e4m3fn);

REGISTER_CAST_GPU(bfloat16, float8_e5m2);
REGISTER_CAST_GPU(bfloat16, float8_e4m3fn);

REGISTER_CAST_GPU(Eigen::half, float8_e5m2);
REGISTER_CAST_GPU(Eigen::half, float8_e4m3fn);

REGISTER_CAST_GPU(float8_e5m2, float);
REGISTER_CAST_GPU(float8_e5m2, bfloat16);
REGISTER_CAST_GPU(float8_e5m2, Eigen::half);
REGISTER_CAST_GPU(float8_e5m2, float8_e5m2);
REGISTER_CAST_GPU(float8_e5m2, float8_e4m3fn);

REGISTER_CAST_GPU(float8_e4m3fn, float);
REGISTER_CAST_GPU(float8_e4m3fn, bfloat16);
REGISTER_CAST_GPU(float8_e4m3fn, Eigen::half);
REGISTER_CAST_GPU(float8_e4m3fn, float8_e5m2);
REGISTER_CAST_GPU(float8_e4m3fn, float8_e4m3fn);

REGISTER_CAST_GPU(int4, int4);
REGISTER_CAST_GPU(int4, int8);
REGISTER_CAST_GPU(int4, int16);
REGISTER_CAST_GPU(int4, int32);
REGISTER_CAST_GPU(int4, int64_t);
REGISTER_CAST_GPU(int4, uint4);
REGISTER_CAST_GPU(int4, uint8);
REGISTER_CAST_GPU(int4, uint16);
REGISTER_CAST_GPU(int4, uint32);
REGISTER_CAST_GPU(int4, uint64_t);

REGISTER_CAST_GPU(int8, int4);
REGISTER_CAST_GPU(int16, int4);
REGISTER_CAST_GPU(int32, int4);
REGISTER_CAST_GPU(int64_t, int4);
REGISTER_CAST_GPU(uint4, int4);
REGISTER_CAST_GPU(uint8, int4);
REGISTER_CAST_GPU(uint16, int4);
REGISTER_CAST_GPU(uint32, int4);
REGISTER_CAST_GPU(uint64_t, int4);

REGISTER_CAST_GPU(uint4, int8);
REGISTER_CAST_GPU(uint4, int16);
REGISTER_CAST_GPU(uint4, int32);
REGISTER_CAST_GPU(uint4, int64_t);
REGISTER_CAST_GPU(uint4, uint4);
REGISTER_CAST_GPU(uint4, uint8);
REGISTER_CAST_GPU(uint4, uint16);
REGISTER_CAST_GPU(uint4, uint32);
REGISTER_CAST_GPU(uint4, uint64_t);

REGISTER_CAST_GPU(int8, uint4);
REGISTER_CAST_GPU(int16, uint4);
REGISTER_CAST_GPU(int32, uint4);
REGISTER_CAST_GPU(int64_t, uint4);
REGISTER_CAST_GPU(uint8, uint4);
REGISTER_CAST_GPU(uint16, uint4);
REGISTER_CAST_GPU(uint32, uint4);
REGISTER_CAST_GPU(uint64_t, uint4);

#undef REGISTER_CAST_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef CURRY_TYPES2

// HostCast differs from Cast in that its input and output are in host memory.
REGISTER_KERNEL_BUILDER(Name("_HostCast").Device(DEVICE_CPU), CpuCastOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostCast").Device(DEVICE_DEFAULT).HostMemory("x").HostMemory("y"),
    CpuCastOp);
}  // end namespace tensorflow
