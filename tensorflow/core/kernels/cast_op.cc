// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/cast_op.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename Tout, typename Tin>
void CastMaybeInline(const Device& d, typename TTypes<Tout>::Flat o,
                     typename TTypes<Tin>::ConstFlat i) {
  if (o.size() * (sizeof(Tin) + sizeof(Tout)) < 131072) {
    // Small cast on a CPU: do inline
    o = i.template cast<Tout>();
  } else {
    o.device(d) = i.template cast<Tout>();
  }
}

template <typename O, typename I>
struct CastFunctor<CPUDevice, O, I> {
  void operator()(const CPUDevice& d, typename TTypes<O>::Flat o,
                  typename TTypes<I>::ConstFlat i) {
    CastMaybeInline<CPUDevice, O, I>(d, o, i);
  }
};

}  // namespace functor

#define CAST_CASE(DEVICE, IN, OUT)                                         \
  if (DataTypeToEnum<IN>::value == src_dtype_ &&                           \
      DataTypeToEnum<OUT>::value == dst_dtype_) {                          \
    work_ = [](OpKernelContext* ctx, const Tensor& inp, Tensor* out) {     \
      functor::CastFunctor<DEVICE, OUT, IN> func;                          \
      func(ctx->eigen_device<DEVICE>(), out->flat<OUT>(), inp.flat<IN>()); \
    };                                                                     \
    return Status::OK();                                                   \
  }

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

  virtual Status Prepare() = 0;
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

 protected:
  Status Prepare() override {
    if (src_dtype_ == dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    CAST_CASE(CPUDevice, bool, float);
    CAST_CASE(CPUDevice, bool, int32);
    CAST_CASE(CPUDevice, bool, double);
    CAST_CASE(CPUDevice, double, float);
    CAST_CASE(CPUDevice, double, int32);
    CAST_CASE(CPUDevice, double, int64);
    CAST_CASE(CPUDevice, float, double);
    CAST_CASE(CPUDevice, float, uint8);
    CAST_CASE(CPUDevice, float, int32);
    CAST_CASE(CPUDevice, float, int64);
    CAST_CASE(CPUDevice, int32, double);
    CAST_CASE(CPUDevice, int32, float);
    CAST_CASE(CPUDevice, int32, uint8);
    CAST_CASE(CPUDevice, int32, int64);
    CAST_CASE(CPUDevice, int64, double);
    CAST_CASE(CPUDevice, int64, float);
    CAST_CASE(CPUDevice, int64, int32);
    CAST_CASE(CPUDevice, uint8, float);
    CAST_CASE(CPUDevice, uint8, int32);
    CAST_CASE(CPUDevice, uint8, int64);
    CAST_CASE(CPUDevice, uint8, double);
    if (src_dtype_ == DT_BFLOAT16 && dst_dtype_ == DT_FLOAT) {
      work_ = [](OpKernelContext* ctx, const Tensor& inp, Tensor* out) {
        int64 N = out->NumElements();
        auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
        int num_threads =
            std::min<int>(std::min(4, worker_threads->num_threads), N / 4096);
        if (num_threads < 1) {
          BFloat16ToFloat(inp.flat<bfloat16>().data(),
                          out->flat<float>().data(), N);
        } else {
          auto work = [&inp, &out](int64 start, int64 end) {
            BFloat16ToFloat(inp.flat<bfloat16>().data() + start,
                            out->flat<float>().data() + start, end - start);
          };
          Shard(num_threads, worker_threads->workers, N, 100, work);
        }
      };
      return Status::OK();
    }
    if (src_dtype_ == DT_FLOAT && dst_dtype_ == DT_BFLOAT16) {
      work_ = [](OpKernelContext* ctx, const Tensor& inp, Tensor* out) {
        int64 N = out->NumElements();
        auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
        int num_threads =
            std::min<int>(std::min(4, worker_threads->num_threads), N / 4096);
        if (num_threads < 1) {
          FloatToBFloat16(inp.flat<float>().data(),
                          out->flat<bfloat16>().data(), N);
        } else {
          auto work = [&inp, &out](int64 start, int64 end) {
            FloatToBFloat16(inp.flat<float>().data() + start,
                            out->flat<bfloat16>().data() + start, end - start);
          };
          Shard(num_threads, worker_threads->workers, N, 100, work);
        }
      };
      return Status::OK();
    }
    return Unimplemented();
  }
};

class GpuCastOp : public CastOpBase {
 public:
  explicit GpuCastOp(OpKernelConstruction* ctx) : CastOpBase(ctx) {
    OP_REQUIRES_OK(ctx, Prepare());
  }

 protected:
  Status Prepare() override {
    if (src_dtype_ == dst_dtype_) {
      work_ = nullptr;  // Identity
      return Status::OK();
    }
    CAST_CASE(GPUDevice, bfloat16, float);
    CAST_CASE(GPUDevice, bool, float);
    CAST_CASE(GPUDevice, double, float);
    CAST_CASE(GPUDevice, double, int64);
    CAST_CASE(GPUDevice, float, bfloat16);
    CAST_CASE(GPUDevice, float, double);
    CAST_CASE(GPUDevice, float, int64);
    CAST_CASE(GPUDevice, int64, double);
    CAST_CASE(GPUDevice, int64, float);
    CAST_CASE(GPUDevice, uint8, float);
    CAST_CASE(GPUDevice, float, uint8);
    CAST_CASE(GPUDevice, bool, int32);
    CAST_CASE(GPUDevice, double, int32);
    CAST_CASE(GPUDevice, float, int32);
    CAST_CASE(GPUDevice, int32, double);
    CAST_CASE(GPUDevice, int32, float);
    CAST_CASE(GPUDevice, int32, int64);
    CAST_CASE(GPUDevice, int64, int32);
    return Unimplemented();
  }
};

#undef CAST_CASE

REGISTER_KERNEL_BUILDER(Name("Cast").Device(DEVICE_CPU), CpuCastOp);

#if GOOGLE_CUDA
#define REGISTER_CAST_GPU(srctype, dsttype)                    \
  REGISTER_KERNEL_BUILDER(Name("Cast")                         \
                              .TypeConstraint<srctype>("SrcT") \
                              .TypeConstraint<dsttype>("DstT") \
                              .Device(DEVICE_GPU),             \
                          GpuCastOp);
REGISTER_CAST_GPU(bfloat16, float);
REGISTER_CAST_GPU(bool, float);
REGISTER_CAST_GPU(double, float);
REGISTER_CAST_GPU(double, int64);
REGISTER_CAST_GPU(float, bfloat16);
REGISTER_CAST_GPU(float, double);
REGISTER_CAST_GPU(float, int64);
REGISTER_CAST_GPU(int64, double);
REGISTER_CAST_GPU(int64, float);
REGISTER_CAST_GPU(uint8, float);
REGISTER_CAST_GPU(float, uint8);
REGISTER_CAST_GPU(bool, int32);
REGISTER_CAST_GPU(double, int32);
REGISTER_CAST_GPU(float, int32);
REGISTER_CAST_GPU(int32, double);
REGISTER_CAST_GPU(int32, float);
REGISTER_CAST_GPU(int32, int64);
REGISTER_CAST_GPU(int64, int32);
#undef REGISTER_CAST_GPU
#endif  // GOOGLE_CUDA

// HostCast differs from Cast in that its input and output are in host memory.
REGISTER_KERNEL_BUILDER(Name("_HostCast").Device(DEVICE_CPU), CpuCastOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostCast").Device(DEVICE_GPU).HostMemory("x").HostMemory("y"),
    CpuCastOp);

}  // end namespace tensorflow
