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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "roll_op.h"

using namespace tensorflow;

#define EIGEN_USE_THREADS
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename T, typename Tshift, typename Taxis>
class RollOp : public OpKernel {
 public:
  explicit RollOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input = context->input(0);
    const Tensor& shift = context->input(1);
    const Tensor& axis = context->input(2);

    // auto input_flat = input.flat<T>();
    auto shift_flat = shift.flat<Tshift>();
    auto axis_flat = axis.flat<Taxis>();


    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(input.shape()),
                errors::InvalidArgument("input must be 1-D or higher"));
    OP_REQUIRES(context, shift.shape().dims() <= 1,
                errors::InvalidArgument("shift must be a scalar or a 1-D vector. Found: ",
                                        shift.shape().DebugString()));
    OP_REQUIRES(context, axis.shape().dims() <= 1,
                errors::InvalidArgument("axis must be a scalar or a 1-D vector. Found: ",
                                        axis.shape().DebugString()));
    OP_REQUIRES(context, shift.shape() == axis.shape(),
                errors::InvalidArgument("shift and axis must be the same size"));
    const int64 N = input.NumElements();
    const int D = static_cast<int>(input.dims());
    const int M = static_cast<int>(shift_flat.size());

    int shifts[D]; // if any duplicate axes, will sum corresponding shifts
    for (int i = 0; i < D; i++) shifts[i] = 0; // default is 0
    for (int i = 0; i < M; i++) {
      const int j = axis_flat(i);
      OP_REQUIRES(context, j < D,
                  errors::InvalidArgument("axis ", j, " is out of range"));
      shifts[j] += static_cast<int>(shift_flat(i));
    }


    int dim_size[D];
    int thresholds[D]; // the index that the roll starts to wrap around
    // dim_stride is the number of indices over in the flattened tensor
    // you need to skip in order to make it over from one side of a dimension
    // to the other. Used to make the shifts wrap around after a threshold.
    int64 dim_strides[D];
    int64 prod_dim_size = 1;
    for (int d = D-1; d >= 0; d--) {
      const int ds = fmax(static_cast<int>(input.dim_size(d)), 1);
      dim_size[d] = ds;
      // modulo that works with negatives: ((x % y) + y) % y
      thresholds[d] = ((ds - shifts[d]) % ds + ds) % ds;
      prod_dim_size *= static_cast<int64>(input.dim_size(d));
      dim_strides[d] = prod_dim_size;
    }

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(),
                                                     &output));


    auto input_flat = input.flat<T>().data();
    auto output_flat = output->flat<T>().data();

    if (std::is_same<Device, CPUDevice>::value) {
      auto work = [input_flat, output_flat, D, &dim_size,
                   &thresholds, &dim_strides](int64 start, int64 end) {
        int in_dim_i[D]; // array of indices for each dimension
        int delta_i = 0; // the difference between out_i and in_i
        // initialize in_dim_i and delta_i
        for (int d = 0; d < D; d++) {
          const int ds = dim_size[d];
          // stride is the number of indices over in the flattened tensor
          // you need to skip in order to make it over to an adjacent element
          // along a dimension.
          const int64 stride = dim_strides[d] / ds;
          // calculated this way will always be positive modulo of shift
          const int shift = ds - thresholds[d];
          const int indx = (start / stride) % ds;
          in_dim_i[d] = indx;
          // calculate dimension index after the shift
          const int out_dim_i = (indx + shift) % ds;
          delta_i += (out_dim_i - indx) * stride;
        }

        for (int64 in_i = start; in_i < end; in_i++) {
          const int64 out_i = in_i + delta_i;
          output_flat[out_i] = input_flat[in_i];

          // create next combination of in_dim_i[d]
          // while at it adjust delta_i if needed
          for (int d = D-1; d >= 0; d--) {
            const int indx = (in_dim_i[d] + 1) % dim_size[d];
            in_dim_i[d] = indx;
            if (indx != 0) {
              if (indx == thresholds[d]) {
                delta_i -= dim_strides[d]; // now wraps around
              }
              break; // indx != 0 don't need to carry
            }else{
              delta_i += dim_strides[d]; // indx became 0 so reverse wrap around
            }
          }
        }
      };
      // Shard
      auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
      const int64 S = fmin(worker_threads->num_threads, N);
      const int64 cost_per_unit = N / fmax(S, 1);
      Shard(worker_threads->num_threads, worker_threads->workers, N, cost_per_unit,
            std::move(work));
      return;
    }


    RollFunctor<Device, T>()(
        context->eigen_device<Device>(),
        N,
        D,
        dim_size,
        input_flat,
        output_flat,
        thresholds,
        dim_strides
    );

  }
};


// Register the CPU kernels.
#define REGISTER_CPU(type)                                        \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_CPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int32>("Tshift")        \
                          .TypeConstraint<int32>("Taxis"),        \
                        RollOp<CPUDevice, type, int32, int32>)    \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_CPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int64>("Tshift")        \
                          .TypeConstraint<int32>("Taxis"),        \
                        RollOp<CPUDevice, type, int64, int32>)    \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_CPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int32>("Tshift")        \
                          .TypeConstraint<int64>("Taxis"),        \
                        RollOp<CPUDevice, type, int32, int64>)    \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_CPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int64>("Tshift")        \
                          .TypeConstraint<int64>("Taxis"),        \
                        RollOp<CPUDevice, type, int64, int64>)

TF_CALL_ALL_TYPES(REGISTER_CPU);
REGISTER_CPU(bfloat16);
#undef REGISTER_CPU


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                           \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_GPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int32>("Tshift")        \
                          .TypeConstraint<int32>("Taxis"),        \
                        RollOp<GPUDevice, type, int32, int32>)    \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_GPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int64>("Tshift")        \
                          .TypeConstraint<int32>("Taxis"),        \
                        RollOp<GPUDevice, type, int64, int32>)    \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_GPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int32>("Tshift")        \
                          .TypeConstraint<int64>("Taxis"),        \
                        RollOp<GPUDevice, type, int32, int64>)    \
REGISTER_KERNEL_BUILDER(Name("Roll")                              \
                          .Device(DEVICE_GPU)                     \
                          .TypeConstraint<type>("T")              \
                          .TypeConstraint<int64>("Tshift")        \
                          .TypeConstraint<int64>("Taxis"),        \
                        RollOp<GPUDevice, type, int64, int64>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU)
#endif  // GOOGLE_CUDA
