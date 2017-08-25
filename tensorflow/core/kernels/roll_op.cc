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
#include "tensorflow/core/framework/register_types_traits.h"
#include "tensorflow/core/util/work_sharder.h"
#include "roll_op.h"

namespace tensorflow {

#define EIGEN_USE_THREADS
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
void DoRoll(OpKernelContext* context, const int64 N,
                const int D, const int* dim_size,
                const T* input, T* output, const int* threshold,
                const int64* dim_range) {

  auto work = [input, output, D, &dim_size,
               &threshold, &dim_range](int64 start, int64 end) {
    int indices[D]; // array of indices for each dimension
    int offset = 0; // the shift along the flat tensor for current element
    // initialize indices and offset
    for (int d = 0; d < D; d++) {
      // stride is the number of indices over in the flattened tensor
      // you need to skip in order to make it over to an adjacent element
      // along a dimension.
      const int64 stride = dim_range[d] / dim_size[d];
      const int shift = dim_size[d] - threshold[d];
      const int indx = (start / stride) % dim_size[d];
      indices[d] = indx;
      // calculate dimension index after the shift
      const int shifted_indx = (indx + shift) % dim_size[d];
      offset += (shifted_indx - indx) * stride;
    }

    for (int64 i = start; i < end; i++) {
      output[i + offset] = input[i];
      // create next combination of indices
      // while at it adjust offset if needed
      for (int d = D-1; d >= 0; d--) {
        const int indx = (indices[d] + 1) % dim_size[d];
        indices[d] = indx;
        if (indx != 0) {
          if (indx == threshold[d]) { // we've reached the threshold
            // dim_range[d] = threshold[d] + shift[d]
            // offset = shift[d] + ... other offsets
            // offset - dim_range[d] = -threshold[d] + ... other offsets
            // thus we undo our previous offset as well as add a new offset of
            // -threshold[d] in one opperation
            offset -= dim_range[d]; // now wraps around
          }
          break; // indx != 0 don't need to carry
        }else if (threshold[d] != 0){ // if threshold is 0 shift is 0
          offset += dim_range[d]; // indx became 0 so reverse wrap around
        }
      }
    }
  };
  // Shard
  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  const int cost_per_unit = 50; // rough esitmate 
  Shard(worker_threads->num_threads, worker_threads->workers, N, cost_per_unit,
        std::move(work));
}

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

    int shift_mod_sum[D]; // if any duplicate axes, will sum corresponding shifts
    for (int d = 0; d < D; d++) shift_mod_sum[d] = 0; // default is 0
    for (int m = 0; m < M; m++) {
      const int a = axis_flat(m);
      OP_REQUIRES(context, a < D,
                  errors::InvalidArgument("axis ", a, " is out of range"));
      const int ds = fmax(static_cast<int>(input.dim_size(a)), 1);
      const int sum = shift_mod_sum[a] + static_cast<int>(shift_flat(m));
      // modulo that works with negatives: ((x % y) + y) % y
      shift_mod_sum[a] = (sum % ds + ds) % ds;
    }

    int dim_size[D];
    int threshold[D]; // the index that the roll starts to wrap around
    // dim_range is the number of indices over in the flattened tensor
    // you need to skip in order to make it over from one side of a dimension
    // to the other. Used to make the shifts wrap around after a threshold.
    int64 dim_range[D];
    int64 dim_size_prod = 1;
    for (int d = D-1; d >= 0; d--) {
      const int ds = fmax(static_cast<int>(input.dim_size(d)), 1);
      dim_size[d] = ds;
      // modulo that works with negatives: ((x % y) + y) % y
      threshold[d] = (ds - shift_mod_sum[d]) % ds;
      dim_size_prod *= static_cast<int64>(input.dim_size(d));
      dim_range[d] = dim_size_prod;
    }

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(),
                                                     &output));
    auto input_flat = input.flat<T>().data();
    auto output_flat = output->flat<T>().data();


    if (std::is_same<Device, CPUDevice>::value) {
      DoRoll<T>(context, N, D, dim_size, input_flat, output_flat,
                     threshold, dim_range);
    }else{ // GPUs and beyond
      RollFunctor<Device, T>()(context->eigen_device<Device>(), N, D, dim_size,
                               input_flat, output_flat, threshold, dim_range);
    }

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
} // namespace tensorflow
