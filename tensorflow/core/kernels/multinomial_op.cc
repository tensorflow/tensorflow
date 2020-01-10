/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/random_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/multinomial_op.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename T, typename OutputType>
struct MultinomialFunctor {
  void operator()(OpKernelContext* ctx, const Device& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<float>::Flat noises,
                  typename TTypes<float>::Flat scores,
                  typename TTypes<float>::Flat scratch, int batch_size,
                  int num_classes, int num_samples,
                  const random::PhiloxRandom& gen,
                  typename TTypes<OutputType>::Matrix output);
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
extern template struct MultinomialFunctor<GPUDevice, Eigen::half, int32>;
extern template struct MultinomialFunctor<GPUDevice, float, int32>;
extern template struct MultinomialFunctor<GPUDevice, double, int32>;
extern template struct MultinomialFunctor<GPUDevice, int32, int32>;
extern template struct MultinomialFunctor<GPUDevice, int64, int32>;

extern template struct MultinomialFunctor<GPUDevice, Eigen::half, int64>;
extern template struct MultinomialFunctor<GPUDevice, float, int64>;
extern template struct MultinomialFunctor<GPUDevice, double, int64>;
extern template struct MultinomialFunctor<GPUDevice, int32, int64>;
extern template struct MultinomialFunctor<GPUDevice, int64, int64>;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T, typename OutputType>
struct MultinomialFunctor<CPUDevice, T, OutputType> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<float>::Flat /* noises */,
                  typename TTypes<float>::Flat /* scores */,
                  typename TTypes<float>::Flat /* scratch */, int batch_size,
                  int num_classes, int num_samples,
                  const random::PhiloxRandom& gen,
                  typename TTypes<OutputType>::Matrix output) {
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    // The implementation only parallelizes by batch.
    //
    // This takes O(BatchSize * NumSamples * log(NumClasses) + NumClasses) CPU
    // time.
    auto DoWork = [ctx, num_samples, num_classes, &gen, &output, &logits](
                      int64 start_row, int64 limit_row) {
      // Capturing "gen" by-value would only make a copy for the _shared_
      // lambda.  Since we want to let each worker have its own copy, we pass
      // "gen" by reference and explicitly do a copy assignment here.
      random::PhiloxRandom gen_copy = gen;
      // Skip takes units of 128 bits.  +3 is so rounding doesn't lead to
      // us using the same state in different batches.
      gen_copy.Skip(start_row * (num_samples + 3) / 4);
      random::SimplePhilox simple_philox(&gen_copy);

      Tensor cdf_tensor;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_DOUBLE, TensorShape({num_classes}),
                                        &cdf_tensor));
      auto cdf = cdf_tensor.flat<double>();
      for (int64 b = start_row; b < limit_row; ++b) {
        const auto* logits_row = &logits(b, 0);

        // Takes an along-class maximum (for numerical stability).
        T max = std::numeric_limits<T>::lowest();
        for (int64 j = 0; j < num_classes; ++j) {
          if (Eigen::numext::isfinite(logits_row[j])) {
            max = std::max(max, logits_row[j]);
          }
        }
        const double max_logit = static_cast<double>(max);

        // Precompute cumulative probability distribution across classes.
        // Note: This isn't normalized.
        cdf = (logits.template chip<0>(b).template cast<double>() - max_logit)
                  .exp();
        double running_total = 0;
        for (int64 j = 0; j < num_classes; ++j) {
          if (Eigen::numext::isfinite(logits_row[j])) {
            running_total += cdf(j);
          }
          cdf(j) = running_total;
        }
        // Generate each sample.
        const double* cdf_begin = cdf.data();
        const double* cdf_end = cdf.data() + num_classes;
        for (int64 j = 0; j < num_samples; ++j) {
          const double to_find = simple_philox.RandDouble() * running_total;
          auto found_iter = std::upper_bound(cdf_begin, cdf_end, to_find);
          output(b, j) = std::distance(cdf_begin, found_iter);
        }
      }
    };
    // Incredibly rough estimate of clock cycles for DoWork();
    const int64 cost =
        50 * (num_samples * std::log(num_classes) / std::log(2) + num_classes);
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost,
          DoWork);
  }
};

}  // namespace functor

namespace {

// Samples from a multinomial distribution.
template <typename Device, typename T, typename OutputType>
class MultinomialOp : public OpKernel {
 public:
  explicit MultinomialOp(OpKernelConstruction* context) : OpKernel(context) {}

  void DoCompute(OpKernelContext* ctx, const Tensor& logits_t,
                 const Tensor& num_samples_t, GuardedPhiloxRandom* generator) {
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits_t.shape()),
                errors::InvalidArgument("logits should be a matrix, got shape ",
                                        logits_t.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(num_samples_t.shape()),
        errors::InvalidArgument("num_samples should be a scalar, got shape ",
                                num_samples_t.shape().DebugString()));

    const int num_samples = num_samples_t.scalar<int>()();
    OP_REQUIRES(ctx, num_samples >= 0,
                errors::InvalidArgument(
                    "num_samples should be nonnegative, got ", num_samples));

    for (int i = 0; i < 2; i++) {
      const int64 dim = logits_t.dim_size(i);
      OP_REQUIRES(ctx, static_cast<int>(dim) == dim,
                  errors::InvalidArgument(
                      "logits.shape = ", logits_t.shape().DebugString(),
                      " too large for int"));
    }
    const int batch_size = static_cast<int>(logits_t.dim_size(0));
    const int num_classes = static_cast<int>(logits_t.dim_size(1));
    OP_REQUIRES(ctx, num_classes > 0,
                errors::InvalidArgument("num_classes should be positive, got ",
                                        num_classes));

    Tensor* samples_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size, num_samples}),
                                  &samples_t));

    // Execute kernel only for nonempty output; otherwise Eigen crashes on GPU.
    if (samples_t->NumElements() > 0) {
      Tensor noises, scores, scratch;  // Scratch space only used for GPU.
      if (std::is_same<Device, GPUDevice>::value) {
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                DT_FLOAT, TensorShape({batch_size, num_samples, num_classes}),
                &noises));
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                DT_FLOAT, TensorShape({batch_size, num_samples, num_classes}),
                &scores));
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(DT_FLOAT, TensorShape({batch_size, num_samples}),
                               &scratch));
      }

      int num_samples_ceil_4 = (num_samples + 3) / 4 * 4;
      // CPU generates doubles = 2 samples per number.
      if (std::is_same<Device, CPUDevice>::value) num_samples_ceil_4 *= 2;
      auto rng =
          generator->ReserveRandomOutputs(batch_size * num_samples_ceil_4, 256);
      functor::MultinomialFunctor<Device, T, OutputType>()(
          ctx, ctx->eigen_device<Device>(), logits_t.matrix<T>(),
          noises.flat<float>(), scores.flat<float>(), scratch.flat<float>(),
          batch_size, num_classes, num_samples, rng,
          samples_t->matrix<OutputType>());
    }
  }
};

template <typename Device, typename T, typename OutputType>
class StatefulMultinomialOp : public MultinomialOp<Device, T, OutputType> {
 public:
  explicit StatefulMultinomialOp(OpKernelConstruction* ctx)
      : MultinomialOp<Device, T, OutputType>(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& logits_t = ctx->input(0);
    const Tensor& num_samples_t = ctx->input(1);
    this->DoCompute(ctx, logits_t, num_samples_t, &generator_);
  }

 private:
  GuardedPhiloxRandom generator_;
};

// TODO(b/77906027): Add a TPU implementation.
#define REGISTER(TYPE)                                                    \
  REGISTER_KERNEL_BUILDER(Name("Multinomial")                             \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<TYPE>("T")                  \
                              .TypeConstraint("output_dtype", DT_INT32),  \
                          StatefulMultinomialOp<CPUDevice, TYPE, int32>); \
  REGISTER_KERNEL_BUILDER(Name("Multinomial")                             \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<TYPE>("T")                  \
                              .TypeConstraint("output_dtype", DT_INT64),  \
                          StatefulMultinomialOp<CPUDevice, TYPE, int64>);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER(TYPE)                                                   \
  REGISTER_KERNEL_BUILDER(Name("Multinomial")                            \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("num_samples")                 \
                              .TypeConstraint<TYPE>("T")                 \
                              .TypeConstraint("output_dtype", DT_INT32), \
                          StatefulMultinomialOp<GPUDevice, TYPE, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Multinomial")                            \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("num_samples")                 \
                              .TypeConstraint<TYPE>("T")                 \
                              .TypeConstraint("output_dtype", DT_INT64), \
                          StatefulMultinomialOp<GPUDevice, TYPE, int64>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T, typename OutputType>
class StatelessMultinomialOp : public MultinomialOp<Device, T, OutputType> {
 public:
  explicit StatelessMultinomialOp(OpKernelConstruction* ctx)
      : MultinomialOp<Device, T, OutputType>(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& logits_t = ctx->input(0);
    const Tensor& num_samples_t = ctx->input(1);

    const Tensor& seed_t = ctx->input(2);
    OP_REQUIRES(ctx, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key, &counter));

    GuardedPhiloxRandom generator;
    generator.Init(counter, key);

    this->DoCompute(ctx, logits_t, num_samples_t, &generator);
  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER(TYPE)                                                     \
  REGISTER_KERNEL_BUILDER(Name("StatelessMultinomial")                     \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<TYPE>("T")                   \
                              .TypeConstraint("output_dtype", DT_INT32),   \
                          StatelessMultinomialOp<CPUDevice, TYPE, int32>); \
  REGISTER_KERNEL_BUILDER(Name("StatelessMultinomial")                     \
                              .Device(DEVICE_CPU)                          \
                              .TypeConstraint<TYPE>("T")                   \
                              .TypeConstraint("output_dtype", DT_INT64),   \
                          StatelessMultinomialOp<CPUDevice, TYPE, int64>);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER(TYPE)                                                    \
  REGISTER_KERNEL_BUILDER(Name("StatelessMultinomial")                    \
                              .Device(DEVICE_GPU)                         \
                              .HostMemory("num_samples")                  \
                              .HostMemory("seed")                         \
                              .TypeConstraint<TYPE>("T")                  \
                              .TypeConstraint("output_dtype", DT_INT32),  \
                          StatelessMultinomialOp<GPUDevice, TYPE, int32>) \
  REGISTER_KERNEL_BUILDER(Name("StatelessMultinomial")                    \
                              .Device(DEVICE_GPU)                         \
                              .HostMemory("num_samples")                  \
                              .HostMemory("seed")                         \
                              .TypeConstraint<TYPE>("T")                  \
                              .TypeConstraint("output_dtype", DT_INT64),  \
                          StatelessMultinomialOp<GPUDevice, TYPE, int64>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace

}  // end namespace tensorflow
