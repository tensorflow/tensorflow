/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/errors.h"
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bincount_op.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using thread::ThreadPool;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Tidx, typename T>
struct BincountFunctor<CPUDevice, Tidx, T, true> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output,
                        const Tidx num_bins) {
    Tensor all_nonneg_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_BOOL, TensorShape({}), &all_nonneg_t, AllocatorAttributes()));
    all_nonneg_t.scalar<bool>().device(context->eigen_cpu_device()) =
        (arr >= Tidx(0)).all();
    if (!all_nonneg_t.scalar<bool>()()) {
      return errors::InvalidArgument("Input arr must be non-negative!");
    }

    // Allocate partial output bin sums for each worker thread. Worker ids in
    // ParallelForWithWorkerId range from 0 to NumThreads() inclusive.
    ThreadPool* thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int64_t num_threads = thread_pool->NumThreads() + 1;
    Tensor partial_bins_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_BOOL, TensorShape({num_threads, num_bins}), &partial_bins_t));
    auto partial_bins = partial_bins_t.matrix<bool>();
    partial_bins.setZero();
    thread_pool->ParallelForWithWorkerId(
        arr.size(), 8 /* cost */,
        [&](int64_t start_ind, int64_t limit_ind, int64_t worker_id) {
          for (int64_t i = start_ind; i < limit_ind; i++) {
            Tidx value = arr(i);
            if (value < num_bins) {
              partial_bins(worker_id, value) = true;
            }
          }
        });

    // Sum the partial bins along the 0th axis.
    Eigen::array<int, 1> reduce_dim({0});
    output.device(context->eigen_cpu_device()) =
        partial_bins.any(reduce_dim).cast<T>();
    return Status::OK();
  }
};

template <typename Tidx, typename T>
struct BincountFunctor<CPUDevice, Tidx, T, false> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output,
                        const Tidx num_bins) {
    Tensor all_nonneg_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_BOOL, TensorShape({}), &all_nonneg_t, AllocatorAttributes()));
    all_nonneg_t.scalar<bool>().device(context->eigen_cpu_device()) =
        (arr >= Tidx(0)).all();
    if (!all_nonneg_t.scalar<bool>()()) {
      return errors::InvalidArgument("Input arr must be non-negative!");
    }

    // Allocate partial output bin sums for each worker thread. Worker ids in
    // ParallelForWithWorkerId range from 0 to NumThreads() inclusive.
    ThreadPool* thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int64_t num_threads = thread_pool->NumThreads() + 1;
    const Tidx* arr_data = arr.data();
    const std::ptrdiff_t arr_size = arr.size();
    const T* weight_data = weights.data();
    if (weights.size() && weights.size() != arr_size) {
      return errors::InvalidArgument(
          "Input indices and weights must have the same size.");
    }
    if (num_threads == 1) {
      output.setZero();
      T* output_data = output.data();
      if (weights.size()) {
        for (int64_t i = 0; i < arr_size; i++) {
          const Tidx value = arr_data[i];
          if (value < num_bins) {
            output_data[value] += weight_data[i];
          }
        }
      } else {
        for (int64_t i = 0; i < arr_size; i++) {
          const Tidx value = arr_data[i];
          if (value < num_bins) {
            // Complex numbers don't support "++".
            output_data[value] += T(1);
          }
        }
      }
    } else {
      Tensor partial_bins_t;
      TF_RETURN_IF_ERROR(context->allocate_temp(
          DataTypeToEnum<T>::value, TensorShape({num_threads, num_bins}),
          &partial_bins_t));
      auto partial_bins = partial_bins_t.matrix<T>();
      partial_bins.setZero();
      thread_pool->ParallelForWithWorkerId(
          arr_size, 8 /* cost */,
          [&](int64_t start_ind, int64_t limit_ind, int64_t worker_id) {
            if (weights.size()) {
              for (int64_t i = start_ind; i < limit_ind; i++) {
                Tidx value = arr_data[i];
                if (value < num_bins) {
                  partial_bins(worker_id, value) += weight_data[i];
                }
              }
            } else {
              for (int64_t i = start_ind; i < limit_ind; i++) {
                Tidx value = arr_data[i];
                if (value < num_bins) {
                  // Complex numbers don't support "++".
                  partial_bins(worker_id, value) += T(1);
                }
              }
            }
          });

      // Sum the partial bins along the 0th axis.
      Eigen::array<int, 1> reduce_dim({0});
      output.device(context->eigen_cpu_device()) = partial_bins.sum(reduce_dim);
    }
    return Status::OK();
  }
};

template <typename Tidx, typename T, bool binary_output>
struct BincountReduceFunctor<CPUDevice, Tidx, T, binary_output> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<Tidx, 2>::ConstTensor& in,
                        const typename TTypes<T, 2>::ConstTensor& weights,
                        typename TTypes<T, 2>::Tensor& out,
                        const Tidx num_bins) {
    const int num_rows = out.dimension(0);
    const int num_cols = in.dimension(1);
    ThreadPool* thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelForWithWorkerId(
        num_rows, 8 /* cost */,
        [&](int64_t start_row, int64_t end_row, int64_t worker_id) {
          for (int64_t i = start_row; i < end_row; ++i) {
            for (int64_t j = 0; j < num_cols; ++j) {
              Tidx value = in(i, j);
              if (value < num_bins) {
                if (binary_output) {
                  out(i, value) = T(1);
                } else {
                  if (weights.size()) {
                    out(i, value) += weights(i, j);
                  } else {
                    out(i, value) += T(1);
                  }
                }
              }
            }
          }
        });
    return Status::OK();
  }
};

}  // namespace functor

template <typename Device, typename T>
class BincountOp : public OpKernel {
 public:
  explicit BincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& arr_t = ctx->input(0);
    const Tensor& size_tensor = ctx->input(1);
    OP_REQUIRES(ctx, size_tensor.dims() == 0,
                errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                        size_tensor.dims()));
    int32_t size = size_tensor.scalar<int32_t>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    const Tensor& weights_t = ctx->input(2);
    const auto arr = arr_t.flat<int32_t>();
    const auto weights = weights_t.flat<T>();
    Tensor* output_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({size}), &output_t));
    auto output = output_t->flat<T>();
    OP_REQUIRES_OK(ctx,
                   functor::BincountFunctor<Device, int32_t, T, false>::Compute(
                       ctx, arr, weights, output, size));
  }
};

#define REGISTER_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Bincount").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BincountOp<CPUDevice, type>)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("Bincount")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("size")         \
                              .TypeConstraint<type>("T"), \
                          BincountOp<GPUDevice, type>)

TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_float(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename Tidx, typename T>
class DenseBincountOp : public OpKernel {
 public:
  explicit DenseBincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("binary_output", &binary_output_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    OP_REQUIRES(ctx, data.dims() <= 2,
                errors::InvalidArgument(
                    "Shape must be at most rank 2 but is rank ", data.dims()));

    const Tensor& size_t = ctx->input(1);
    const Tensor& weights = ctx->input(2);

    Tidx size = size_t.scalar<Tidx>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    Tensor* out_t;
    functor::SetZeroFunctor<Device, T> fill;
    if (data.dims() == 1) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({size}), &out_t));
      auto out = out_t->flat<T>();
      fill(ctx->eigen_device<Device>(), out);
      if (binary_output_) {
        OP_REQUIRES_OK(
            ctx, functor::BincountFunctor<Device, Tidx, T, true>::Compute(
                     ctx, data.flat<Tidx>(), weights.flat<T>(), out, size));
      } else {
        OP_REQUIRES_OK(
            ctx, functor::BincountFunctor<Device, Tidx, T, false>::Compute(
                     ctx, data.flat<Tidx>(), weights.flat<T>(), out, size));
      }
    } else if (data.dims() == 2) {
      const int64_t num_rows = data.dim_size(0);
      auto weight_matrix =
          (weights.NumElements() == 0)
              ? weights.shaped<T, 2>(gtl::InlinedVector<int64_t, 2>(2, 0))
              : weights.matrix<T>();
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, TensorShape({num_rows, size}), &out_t));
      auto out = out_t->matrix<T>();
      fill(ctx->eigen_device<Device>(), out_t->flat<T>());
      if (binary_output_) {
        OP_REQUIRES_OK(
            ctx, functor::BincountReduceFunctor<Device, Tidx, T, true>::Compute(
                     ctx, data.matrix<Tidx>(), weight_matrix, out, size));
      } else {
        OP_REQUIRES_OK(
            ctx,
            functor::BincountReduceFunctor<Device, Tidx, T, false>::Compute(
                ctx, data.matrix<Tidx>(), weight_matrix, out, size));
      }
    }
  }

 private:
  bool binary_output_;
};

#define REGISTER_KERNELS(Tidx, T)                            \
  REGISTER_KERNEL_BUILDER(Name("DenseBincount")              \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<Tidx>("Tidx"), \
                          DenseBincountOp<CPUDevice, Tidx, T>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64, T);

TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_KERNELS(Tidx, T)                            \
  REGISTER_KERNEL_BUILDER(Name("DenseBincount")              \
                              .Device(DEVICE_GPU)            \
                              .HostMemory("size")            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<Tidx>("Tidx"), \
                          DenseBincountOp<GPUDevice, Tidx, T>);
#define REGISTER_GPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64, T);

TF_CALL_int32(REGISTER_GPU_KERNELS);
TF_CALL_float(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename Tidx, typename T>
class SparseBincountOp : public OpKernel {
 public:
  explicit SparseBincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("binary_output", &binary_output_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices = ctx->input(0);
    const auto values = ctx->input(1).flat<Tidx>();
    const Tensor& dense_shape = ctx->input(2);
    const Tensor& size_t = ctx->input(3);
    const auto weights = ctx->input(4).flat<T>();
    const int64_t weights_size = weights.size();

    Tidx size = size_t.scalar<Tidx>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    bool is_1d = dense_shape.NumElements() == 1;

    Tensor* out_t;
    functor::SetZeroFunctor<Device, T> fill;
    if (is_1d) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({size}), &out_t));
      auto out = out_t->flat<T>();
      fill(ctx->eigen_device<Device>(), out);
      if (binary_output_) {
        OP_REQUIRES_OK(ctx,
                       functor::BincountFunctor<Device, Tidx, T, true>::Compute(
                           ctx, values, weights, out, size));
      } else {
        OP_REQUIRES_OK(
            ctx, functor::BincountFunctor<Device, Tidx, T, false>::Compute(
                     ctx, values, weights, out, size));
      }
    } else {
      const auto shape = dense_shape.flat<int64_t>();
      const int64_t num_rows = shape(0);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, TensorShape({num_rows, size}), &out_t));
      const auto out = out_t->matrix<T>();
      fill(ctx->eigen_device<Device>(), out_t->flat<T>());
      const auto indices_mat = indices.matrix<int64_t>();
      for (int64_t i = 0; i < indices_mat.dimension(0); ++i) {
        const int64_t batch = indices_mat(i, 0);
        const Tidx bin = values(i);
        if (bin < size) {
          if (binary_output_) {
            out(batch, bin) = T(1);
          } else {
            if (weights_size) {
              out(batch, bin) += weights(i);
            } else {
              out(batch, bin) += T(1);
            }
          }
        }
      }
    }
  }

 private:
  bool binary_output_;
};

#define REGISTER_KERNELS(Tidx, T)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseBincount")             \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<Tidx>("Tidx"), \
                          SparseBincountOp<CPUDevice, Tidx, T>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64, T);

TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename Tidx, typename T>
class RaggedBincountOp : public OpKernel {
 public:
  explicit RaggedBincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("binary_output", &binary_output_));
  }

  void Compute(OpKernelContext* ctx) override {
    const auto splits = ctx->input(0).flat<int64_t>();
    const auto values = ctx->input(1).flat<Tidx>();
    const Tensor& size_t = ctx->input(2);
    const auto weights = ctx->input(3).flat<T>();
    const int64_t weights_size = weights.size();

    Tidx size = size_t.scalar<Tidx>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    int num_rows = splits.size() - 1;
    int num_values = values.size();
    int batch_idx = 0;

    OP_REQUIRES(ctx, splits(0) == 0,
                errors::InvalidArgument("Splits must start with 0, not with ",
                                        splits(0)));

    OP_REQUIRES(ctx, splits(num_rows) == num_values,
                errors::InvalidArgument(
                    "Splits must end with the number of values, got ",
                    splits(num_rows), " instead of ", num_values));

    Tensor* out_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({num_rows, size}), &out_t));
    functor::SetZeroFunctor<Device, T> fill;
    fill(ctx->eigen_device<Device>(), out_t->flat<T>());
    const auto out = out_t->matrix<T>();

    for (int idx = 0; idx < num_values; ++idx) {
      while (idx >= splits(batch_idx)) {
        batch_idx++;
      }
      Tidx bin = values(idx);
      OP_REQUIRES(ctx, bin >= 0,
                  errors::InvalidArgument("Input must be non-negative"));
      if (bin < size) {
        if (binary_output_) {
          out(batch_idx - 1, bin) = T(1);
        } else {
          T value = (weights_size > 0) ? weights(idx) : T(1);
          out(batch_idx - 1, bin) += value;
        }
      }
    }
  }

 private:
  bool binary_output_;
};

#define REGISTER_KERNELS(Tidx, T)                            \
  REGISTER_KERNEL_BUILDER(Name("RaggedBincount")             \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T")        \
                              .TypeConstraint<Tidx>("Tidx"), \
                          RaggedBincountOp<CPUDevice, Tidx, T>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(int32, T);   \
  REGISTER_KERNELS(int64, T);

TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // end namespace tensorflow
