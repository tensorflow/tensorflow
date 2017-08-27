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

#include "roll_op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/register_types_traits.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

#define EIGEN_USE_THREADS
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
void DoRoll(OpKernelContext* context, const int64 N, const int D,
            const int* dim_size, const T* input, T* output,
            const int* threshold, const int64* dim_range) {
  auto work = [input, output, D, &dim_size, &threshold, &dim_range](int64 start,
                                                                    int64 end) {
    int indices[D];  // array of indices for each dimension
    int offset = 0;  // the shift along the flattened tensor for current element
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
      for (int d = D - 1; d >= 0; d--) {
        const int indx = (indices[d] + 1) % dim_size[d];
        indices[d] = indx;
        if (indx != 0) {
          if (indx == threshold[d]) {  // we've reached the threshold
            // dim_range[d] = threshold[d] + shift[d]
            // offset = shift[d] + ... other offsets
            // offset - dim_range[d] = -threshold[d] + ... other offsets
            // thus we undo our previous offset as well as add a new offset of
            // -threshold[d] in one opperation
            offset -= dim_range[d];  // now wraps around
          }
          break;                         // indx != 0 don't need to carry
        } else if (threshold[d] != 0) {  // if threshold is 0 shift is 0
          offset += dim_range[d];        // indx became 0 so reverse wrap around
        }
      }
    }
  };
  // Shard
  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  const int cost_per_unit = 50;  // rough esitmate
  Shard(worker_threads->num_threads, worker_threads->workers, N, cost_per_unit,
        std::move(work));
}

template <typename T>
// Use memcpy to copy memory in groups when the data type supports memcpy
void DoRollV2(OpKernelContext* context, const int64 N, const int D,
              const int* dim_size, const T* input, T* output,
              const int* threshold, const int64* dim_range, const int64 isd) {
  auto work = [input, output, D, &dim_size, &threshold, &dim_range, isd](
                  int64 start, int64 end) {
    const T* in_ptr = &input[0];
    T* out_ptr = &output[0];
    in_ptr += start;
    out_ptr += start;

    // array of indices for each dimension
    // indicies = [i, j, k, l, m, n]
    int indicies[D];
    // the offset needed to make all inner non-shifting dimensions become 0
    int64 remainder_offset = 0;
    // initialize indicies
    for (int d = 0; d < D; d++) {
      // stride is the number of indices over in the flattened tensor
      // you need to skip in order to make it over to an adjacent element
      // along a dimension.
      const int64 stride = dim_range[d] / dim_size[d];
      const int shift = dim_size[d] - threshold[d];
      const int indx = (start / stride) % dim_size[d];
      indicies[d] = indx;
      // calculate dimension index after the shift
      int out_indx = (indx + shift) % dim_size[d];
      if (d > isd) {
        // trailing zeroes for indices after the inner shifted dimension
        out_indx = 0;
        remainder_offset += (out_indx - indx) * stride;
      }
      out_ptr += (out_indx - indx) * stride;
    }
    // set trailing zeroes for indices after the inner shifted dimension
    for (int d = D - 1; d > isd; d--) indicies[d] = 0;
    // the distance along the flattend tensor to the next element in the isd
    const int64 isd_stride = dim_range[isd] / dim_size[isd];

    // the number of indices in the isd dimension the next group will skip
    // to make it to the next threshold or end point
    int isd_indx_skip = 0;
    // the size of the next group
    int64 group_size = 0;
    // initialize isd_indx_skip and group_size
    if (indicies[isd] < threshold[isd]) {
      isd_indx_skip = threshold[isd] - indicies[isd];
      group_size = isd_indx_skip * isd_stride + remainder_offset;
    } else {
      isd_indx_skip = dim_size[isd] - indicies[isd];
      group_size = isd_indx_skip * isd_stride + remainder_offset;
    }

    int64 i = start;
    while (i < end) {
      // copy group of elements
      memcpy(out_ptr, in_ptr, group_size * sizeof(T));

      // shift i and the pointers over to the next group position
      i += group_size;
      out_ptr += group_size;
      in_ptr += group_size;

      // produce next combination of indices and adjust the out_ptr position
      // to fix the offset if necessary
      // the isd should skip to next threshold or endpoint
      // all dimensions to the left increment by 1 when a digit is carried
      // all dimensions to the right remain set to 0
      //            +1 +1 +1 +isd_indx_skip
      // indicies = [i, j, k, l, 0, 0]
      //                      ^isd
      for (int d = isd; d >= 0; d--) {
        int inc = 1;
        if (d == isd) inc = isd_indx_skip;
        const int indx = (indicies[d] + inc) % dim_size[d];
        indicies[d] = indx;
        if (indx != 0) {
          if (indx == threshold[d]) {
            out_ptr -= dim_range[d];  // now wraps around
          }
          break;  // indx != 0 don't need to carry
        } else {
          out_ptr += dim_range[d];  // indx became 0 so reverse wrap around
        }
      }

      // set isd_indx_skip and group_size for next iteration
      if (indicies[isd] < threshold[isd]) {
        isd_indx_skip = threshold[isd] - indicies[isd];
        group_size = isd_indx_skip * isd_stride;
      } else {
        isd_indx_skip = dim_size[isd] - indicies[isd];
        group_size = isd_indx_skip * isd_stride;
      }
    }
  };
  // Shard
  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  const int64 ave_group_size = dim_range[isd] / 2;
  const int cost_per_unit = 50 / fmax(ave_group_size, 1);  // rough esitmate
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
                errors::InvalidArgument(
                    "shift must be a scalar or a 1-D vector. Found: ",
                    shift.shape().DebugString()));
    OP_REQUIRES(context, axis.shape().dims() <= 1,
                errors::InvalidArgument(
                    "axis must be a scalar or a 1-D vector. Found: ",
                    axis.shape().DebugString()));
    OP_REQUIRES(
        context, shift.shape() == axis.shape(),
        errors::InvalidArgument("shift and axis must be the same size"));
    const int64 N = input.NumElements();
    const int M = static_cast<int>(shift_flat.size());
    const int D = static_cast<int>(input.dims());

    int shift_mod_sum[D];  // if any duplicate axes, will sum corresponding
                           // shifts
    for (int d = 0; d < D; d++) shift_mod_sum[d] = 0;  // default is 0
    for (int m = 0; m < M; m++) {
      const int a = axis_flat(m);
      OP_REQUIRES(context, a < D,
                  errors::InvalidArgument("axis ", a, " is out of range"));
      const int ds = fmax(static_cast<int>(input.dim_size(a)), 1);
      const int sum = shift_mod_sum[a] + static_cast<int>(shift_flat(m));
      // modulo that works with negatives: ((x % y) + y) % y
      shift_mod_sum[a] = (sum % ds + ds) % ds;
    }
    // the size of each dimension
    int dim_size[D];
    // threshold[d] is the index that the roll starts to wrap back to the front
    int threshold[D];
    // dim_range is the number of indices over in the flattened tensor
    // you need to skip in order to make it over from one side of a dimension
    // to the other. Used to make the shifts wrap around after a threshold.
    int64 dim_range[D];
    int64 dim_size_prod = 1;
    // inner shift dimension (inner most shifted dimension)
    int64 isd = 0;
    for (int d = D - 1; d >= 0; d--) {
      if (!isd && shift_mod_sum[d]) isd = d;
      const int ds = fmax(static_cast<int>(input.dim_size(d)), 1);
      dim_size[d] = ds;
      threshold[d] = (ds - shift_mod_sum[d]) % ds;
      dim_size_prod *= static_cast<int64>(input.dim_size(d));
      dim_range[d] = dim_size_prod;
    }

    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    auto input_flat = input.flat<T>().data();
    auto output_flat = output->flat<T>().data();

    if (std::is_same<Device, CPUDevice>::value || N > kint32max) {
      // if N > kint32max this is too large for GPUs so we'll use CPU
      if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
        // V2 copies memory in groups instead of element by element
        // much faster
        DoRollV2<T>(context, N, D, dim_size, input_flat, output_flat, threshold,
                    dim_range, isd);
      } else {
        // incase memcpy does not work for current data type
        DoRoll<T>(context, N, D, dim_size, input_flat, output_flat, threshold,
                  dim_range);
      }
    } else {
      // for GPUs
      RollFunctor<Device, T>()(context->eigen_device<Device>(), N, D, dim_size,
                               input_flat, output_flat, threshold, dim_range);
    }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(type)                                       \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int32>("Taxis"),   \
                          RollOp<CPUDevice, type, int32, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int32>("Taxis"),   \
                          RollOp<CPUDevice, type, int64, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int64>("Taxis"),   \
                          RollOp<CPUDevice, type, int32, int64>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int64>("Taxis"),   \
                          RollOp<CPUDevice, type, int64, int64>)

TF_CALL_ALL_TYPES(REGISTER_CPU);
REGISTER_CPU(bfloat16);
#undef REGISTER_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int32>("Taxis"),   \
                          RollOp<GPUDevice, type, int32, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int32>("Taxis"),   \
                          RollOp<GPUDevice, type, int64, int32>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("Tshift")   \
                              .TypeConstraint<int64>("Taxis"),   \
                          RollOp<GPUDevice, type, int32, int64>) \
  REGISTER_KERNEL_BUILDER(Name("Roll")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("Tshift")   \
                              .TypeConstraint<int64>("Taxis"),   \
                          RollOp<GPUDevice, type, int64, int64>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU)
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
