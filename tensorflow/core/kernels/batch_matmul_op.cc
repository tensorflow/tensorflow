// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Scalar>
struct LaunchBatchMatMul;

template <typename Scalar>
struct LaunchBatchMatMul<CPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y, Tensor* out) {
    auto Tx = in_x.tensor<Scalar, 3>();
    auto Ty = in_y.tensor<Scalar, 3>();
    auto Tz = out->tensor<Scalar, 3>();

    // Shards "n"-matmuls into "num" shards. Each shard is
    // dispatched to a thread.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    const int64 num_units = in_x.dim_size(0);
    const int64 cost_per_unit =
        in_x.dim_size(0) * in_x.dim_size(1) * out->dim_size(2);
    Shard(worker_threads.num_threads, worker_threads.workers, num_units,
          cost_per_unit, [&Tx, &Ty, adj_x, adj_y, &Tz](int start, int limit) {
            LaunchBatchMatMul<CPUDevice, Scalar>::Run(Tx, Ty, adj_x, adj_y, Tz,
                                                      start, limit);
          });
  }

  template <typename In, typename Out>
  static void Run(In Tx, In Ty, bool adj_x, bool adj_y, Out Tz, int start,
                  int limit) {
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;

    Eigen::internal::scalar_conjugate_op<Scalar> conj;
    if (!adj_x && !adj_y) {
      for (int i = start; i < limit; ++i) {
        auto x = Tx.template chip<0>(i);
        auto y = Ty.template chip<0>(i);
        auto z = Tz.template chip<0>(i);
        contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
        z = x.contract(y, contract_pairs);  // matmul
      }
    } else if (!adj_x && adj_y) {
      for (int i = start; i < limit; ++i) {
        auto x = Tx.template chip<0>(i);
        auto y = Ty.template chip<0>(i).unaryExpr(conj);
        auto z = Tz.template chip<0>(i);
        contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 1);
        z = x.contract(y, contract_pairs);  // matmul
      }
    } else if (adj_x && !adj_y) {
      for (int i = start; i < limit; ++i) {
        auto x = Tx.template chip<0>(i).unaryExpr(conj);
        auto y = Ty.template chip<0>(i);
        auto z = Tz.template chip<0>(i);
        contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(0, 0);
        z = x.contract(y, contract_pairs);  // matmul
      }
    } else {
      for (int i = start; i < limit; ++i) {
        auto x = Tx.template chip<0>(i).unaryExpr(conj);
        auto y = Ty.template chip<0>(i).unaryExpr(conj);
        auto z = Tz.template chip<0>(i);
        contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(0, 1);
        z = x.contract(y, contract_pairs);  // matmul
      }
    }
  }
};

#if GOOGLE_CUDA

namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

template <typename Scalar>
struct LaunchBatchMatMul<GPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y, Tensor* out) {
    perftools::gputools::blas::Transpose trans[] = {
        perftools::gputools::blas::Transpose::kNoTranspose,
        perftools::gputools::blas::Transpose::kTranspose};
    const uint64 m = in_x.dim_size(adj_x ? 2 : 1);
    const uint64 k = in_x.dim_size(adj_x ? 1 : 2);
    const uint64 n = in_y.dim_size(adj_y ? 1 : 2);
    const uint64 batch_size = in_x.dim_size(0);
    auto blas_transpose_a = trans[adj_x];
    auto blas_transpose_b = trans[adj_y];

    auto* stream = context->op_device_context<GPUDeviceContext>()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    typedef perftools::gputools::DeviceMemory<Scalar> DeviceMemoryType;
    std::vector<DeviceMemoryType> a_device_memory;
    std::vector<DeviceMemoryType> b_device_memory;
    std::vector<DeviceMemoryType> c_device_memory;
    std::vector<DeviceMemoryType*> a_ptrs;
    std::vector<DeviceMemoryType*> b_ptrs;
    std::vector<DeviceMemoryType*> c_ptrs;
    a_device_memory.reserve(batch_size);
    b_device_memory.reserve(batch_size);
    c_device_memory.reserve(batch_size);
    a_ptrs.reserve(batch_size);
    b_ptrs.reserve(batch_size);
    c_ptrs.reserve(batch_size);
    auto* a_base_ptr = in_x.template flat<Scalar>().data();
    auto* b_base_ptr = in_y.template flat<Scalar>().data();
    auto* c_base_ptr = out->template flat<Scalar>().data();
    for (int64 i = 0; i < batch_size; ++i) {
      a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
      b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
      c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());
    }

    // Cublas does
    // C = A x B
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A' (' stands for transpose)
    bool blas_launch_status =
        stream->ThenBlasGemmBatched(blas_transpose_b, blas_transpose_a, n, m, k,
                                    static_cast<Scalar>(1.0), b_ptrs,
                                    adj_y ? k : n, a_ptrs, adj_x ? m : k,
                                    static_cast<Scalar>(0.0), c_ptrs, n,
                                    batch_size)
            .ok();
    if (!blas_launch_status) {
      context->SetStatus(errors::Internal(
          "Blas SGEMMBatched launch failed : a.shape=",
          in_x.shape().DebugString(), ", b.shape=", in_y.shape().DebugString(),
          ", m=", m, ", n=", n, ", k=", k, ", batch_size=", batch_size));
    }
  }
};

#endif  // GOOGLE_CUDA

template <typename Device, typename Scalar>
class BatchMatMul : public OpKernel {
 public:
  explicit BatchMatMul(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  virtual ~BatchMatMul() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    OP_REQUIRES(ctx, in0.dims() == in1.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        in0.shape().ShortDebugString(), " vs. ",
                                        in1.shape().ShortDebugString()));
    const int ndims = in0.dims();
    OP_REQUIRES(
        ctx, ndims >= 3,
        errors::InvalidArgument("In[0] and In[1] ndims must be >= 3: ", ndims));
    TensorShape out_shape;
    for (int i = 0; i < ndims - 2; ++i) {
      OP_REQUIRES(ctx, in0.dim_size(i) == in1.dim_size(i),
                  errors::InvalidArgument("In[0].dim(", i, ") and In[1].dim(",
                                          i, ") must be the same: ",
                                          in0.shape().DebugString(), " vs ",
                                          in1.shape().DebugString()));
      out_shape.AddDim(in0.dim_size(i));
    }
    auto n = out_shape.num_elements();
    auto d0 = in0.dim_size(ndims - 2);
    auto d1 = in0.dim_size(ndims - 1);
    Tensor in0_reshaped;
    CHECK(in0_reshaped.CopyFrom(in0, TensorShape({n, d0, d1})));
    auto d2 = in1.dim_size(ndims - 2);
    auto d3 = in1.dim_size(ndims - 1);
    Tensor in1_reshaped;
    CHECK(in1_reshaped.CopyFrom(in1, TensorShape({n, d2, d3})));
    if (adj_x_) std::swap(d0, d1);
    if (adj_y_) std::swap(d2, d3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().ShortDebugString(), " ",
                    in1.shape().ShortDebugString(), " ", adj_x_, " ", adj_y_));
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (in0.NumElements() == 0 || in1.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }
    Tensor out_reshaped;
    CHECK(out_reshaped.CopyFrom(*out, TensorShape({n, d0, d3})));
    LaunchBatchMatMul<Device, Scalar>::Launch(ctx, in0_reshaped, in1_reshaped,
                                              adj_x_, adj_y_, &out_reshaped);
  }

 private:
  bool adj_x_;
  bool adj_y_;
};

#define REGISTER_CPU(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      BatchMatMul<CPUDevice, TYPE>)

#define REGISTER_GPU(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BatchMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      BatchMatMul<GPUDevice, TYPE>)

REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(int32);
REGISTER_CPU(complex64);

#ifdef GOOGLE_CUDA
// TODO(kalakris): The GPU implementation is currently disabled due to issues
// encountered in practice. See b/24534272.
// REGISTER_GPU(float);
#endif  // GOOGLE_CUDA

#undef REGISTER_CPU
#undef REGISTER_GPU
}  // end namespace tensorflow
