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

#define CURRY_TYPES2(FN, arg0)   \
  FN(arg0, bool);                \
  FN(arg0, uint8);               \
  FN(arg0, int8);                \
  FN(arg0, uint16);              \
  FN(arg0, int16);               \
  FN(arg0, int32);               \
  FN(arg0, int64);               \
  FN(arg0, Eigen::half);         \
  FN(arg0, float);               \
  FN(arg0, double);              \
  FN(arg0, std::complex<float>); \
  FN(arg0, std::complex<double>)

class CastOpBase : public OpKernel {
 public:
  explicit CastOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("SrcT", &src_dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DstT", &dst_dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    if (work_ == nullptr) {
      ctx->set_output(0, inp);
    } else {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &out));
      work_(ctx, inp, out);
    }
  }

 protected:
  DataType src_dtype_;
  DataType dst_dtype_;
  std::function<void(OpKernelContext*, const Tensor&, Tensor*)> work_ = nullptr;

  Status Unimplemented() {
    return errors::Unimplemented("Cast ", DataTypeString(src_dtype_), " to ",
                                 DataTypeString(dst_dtype_),
                                 " is not supported");
  }

  TF_DISALLOW_COPY_AND_ASSIGN(CastOpBase);
};

class CpuCastOp : public CastOpBase {
 public:
  explicit CpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 private:
  Status Prepare() {
    if (src_dtype_ == dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    if (src_dtype_ == DT_BOOL) {
      work_ = GetCpuCastFromBool(dst_dtype_);
    } else if (src_dtype_ == DT_UINT8) {
      work_ = GetCpuCastFromUint8(dst_dtype_);
    } else if (src_dtype_ == DT_INT8) {
      work_ = GetCpuCastFromInt8(dst_dtype_);
    } else if (src_dtype_ == DT_UINT16) {
      work_ = GetCpuCastFromUint16(dst_dtype_);
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
};

#if GOOGLE_CUDA
class GpuCastOp : public CastOpBase {
 public:
  explicit GpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 private:
  Status Prepare() {
    if (src_dtype_ == dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    if (src_dtype_ == DT_BOOL) {
      work_ = GetGpuCastFromBool(dst_dtype_);
    } else if (src_dtype_ == DT_UINT8) {
      work_ = GetGpuCastFromUint8(dst_dtype_);
    } else if (src_dtype_ == DT_INT8) {
      work_ = GetGpuCastFromInt8(dst_dtype_);
    } else if (src_dtype_ == DT_UINT16) {
      work_ = GetGpuCastFromUint16(dst_dtype_);
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
#endif  // GOOGLE_CUDA

#undef CAST_CASE

REGISTER_KERNEL_BUILDER(Name("Cast").Device(DEVICE_CPU), CpuCastOp);

#if GOOGLE_CUDA
#define REGISTER_CAST_GPU(srctype, dsttype)                    \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_GPU),             \
                          GpuCastOp)

CURRY_TYPES2(REGISTER_CAST_GPU, bool);
CURRY_TYPES2(REGISTER_CAST_GPU, uint8);
CURRY_TYPES2(REGISTER_CAST_GPU, int8);
CURRY_TYPES2(REGISTER_CAST_GPU, uint16);
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
#endif  // GOOGLE_CUDA

#undef CURRY_TYPES2

// HostCast differs from Cast in that its input and output are in host memory.
REGISTER_KERNEL_BUILDER(Name("_HostCast").Device(DEVICE_CPU), CpuCastOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostCast").Device(DEVICE_GPU).HostMemory("x").HostMemory("y"),
    CpuCastOp);

}  // end namespace tensorflow
