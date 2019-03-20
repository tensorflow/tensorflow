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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/kernels/cast_op_impl.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

#define CURRY_TYPES2(FN, arg0)   \
  FN(arg0, bool);                \
  FN(arg0, uint8);               \
  FN(arg0, uint16);              \
  FN(arg0, uint32);              \
  FN(arg0, uint64);              \
  FN(arg0, int8);                \
  FN(arg0, int16);               \
  FN(arg0, int32);               \
  FN(arg0, int64);               \
  FN(arg0, Eigen::half);         \
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
  } else {
    Tensor in;
    if (external_src_dtype_ != src_dtype_) {
      // If the type is a quantized type we need to do a bitcast since the
      // src_dtype_ is different from external_src_type_.
      OP_REQUIRES_OK(ctx, in.BitcastFrom(inp, src_dtype_, inp.shape()));
    } else {
      in = inp;
    }
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &out));
    out->set_dtype(dst_dtype_);
    work_(ctx, in, out, use_truncation_);
    out->set_dtype(external_dst_dtype_);
  }
}

Status CastOpBase::Unimplemented() {
  return errors::Unimplemented("Cast ", DataTypeString(external_src_dtype_),
                               " to ", DataTypeString(external_dst_dtype_),
                               " is not supported");
}

CpuCastOp::CpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
  OP_REQUIRES_OK(ctx, Prepare());
}

Status CpuCastOp::Prepare() {
  if (external_src_dtype_ == external_dst_dtype_) {
    work_ = nullptr;  // Identity
    return Status::OK();
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
  }

  // TODO(sesse): If CPU casting to or from Eigen::half ever becomes a
  // bottleneck, we could probably implement specialized support for
  // vectorized versions (not the least based on F16C for Haswell
  // or newer).

  return work_ == nullptr ? Unimplemented() : Status::OK();
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
class GpuCastOp : public CastOpBase {
 public:
  explicit GpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 private:
  Status Prepare() {
    if (external_src_dtype_ == external_dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
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
    }

    return work_ == nullptr ? Unimplemented() : Status::OK();
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef CAST_CASE

REGISTER_KERNEL_BUILDER(Name("Cast").Device(DEVICE_CPU), CpuCastOp);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_CAST_GPU(srctype, dsttype)                    \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_GPU),             \
                          GpuCastOp)

CURRY_TYPES2(REGISTER_CAST_GPU, bool);
CURRY_TYPES2(REGISTER_CAST_GPU, uint8);
CURRY_TYPES2(REGISTER_CAST_GPU, uint16);
CURRY_TYPES2(REGISTER_CAST_GPU, uint32);
CURRY_TYPES2(REGISTER_CAST_GPU, uint64);
CURRY_TYPES2(REGISTER_CAST_GPU, int8);
CURRY_TYPES2(REGISTER_CAST_GPU, int16);
CURRY_TYPES2(REGISTER_CAST_GPU, int32);
CURRY_TYPES2(REGISTER_CAST_GPU, int64);
CURRY_TYPES2(REGISTER_CAST_GPU, Eigen::half);
CURRY_TYPES2(REGISTER_CAST_GPU, float);
CURRY_TYPES2(REGISTER_CAST_GPU, double);
CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<float>);
CURRY_TYPES2(REGISTER_CAST_GPU, std::complex<double>);
REGISTER_CAST_GPU(float, bfloat16);
REGISTER_CAST_GPU(bfloat16, float);

#undef REGISTER_CAST_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
class SyclCastOp : public CastOpBase {
 public:
  explicit SyclCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 private:
  Status Prepare() {
    if (external_src_dtype_ == external_dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    if (src_dtype_ == DT_BOOL) {
      work_ = GetSyclCastFromBool(dst_dtype_);
    } else if (src_dtype_ == DT_INT32) {
      work_ = GetSyclCastFromInt32(dst_dtype_);
    } else if (src_dtype_ == DT_INT64) {
      work_ = GetSyclCastFromInt64(dst_dtype_);
    } else if (src_dtype_ == DT_FLOAT) {
      work_ = GetSyclCastFromFloat(dst_dtype_);
    } else if (src_dtype_ == DT_DOUBLE) {
      work_ = GetSyclCastFromDouble(dst_dtype_);
    }

    return work_ == nullptr ? Unimplemented() : Status::OK();
  }
};

#define REGISTER_CAST_SYCL(srctype, dsttype)                   \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_SYCL),            \
                          SyclCastOp)
CURRY_TYPES2(REGISTER_CAST_SYCL, bool);
CURRY_TYPES2(REGISTER_CAST_SYCL, int32);
CURRY_TYPES2(REGISTER_CAST_SYCL, int64);
CURRY_TYPES2(REGISTER_CAST_SYCL, float);
CURRY_TYPES2(REGISTER_CAST_SYCL, double);

#undef REGISTER_CAST_SYCL

#endif  // TENSORFLOW_USE_SYCL

#undef CURRY_TYPES2

// HostCast differs from Cast in that its input and output are in host memory.
REGISTER_KERNEL_BUILDER(Name("_HostCast").Device(DEVICE_CPU), CpuCastOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostCast").Device(DEVICE_GPU).HostMemory("x").HostMemory("y"),
    CpuCastOp);
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("_HostCast").Device(DEVICE_SYCL).HostMemory("x").HostMemory("y"),
    CpuCastOp);
#endif  // TENSORFLOW_USE_SYCL
}  // end namespace tensorflow
