/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/register_types_traits.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

#define EIGEN_USE_THREADS
using CPUDevice = Eigen::ThreadPoolDevice;

// dim_size - the size of each dimension
// dim_range - the number of indices over in the flattened tensor
//    you need to skip in order to make it over from one side of a dimension
//    to the other. Used to make the shifts wrap around after a threshold.
// threshold - the index for each dimension that the roll starts to wrap
//    back to the front
template <typename T>
void DoRoll(OpKernelContext* context, const int64 num_elements,
            const int num_dims, const gtl::ArraySlice<int>& dim_size,
            const T* input, T* output, const gtl::ArraySlice<int>& threshold,
            const gtl::ArraySlice<int64>& dim_range) {
  auto work = [input, output, num_dims, &dim_size, &threshold, &dim_range](
                  int64 start, int64 end) {
    // array of indices for each dimension
    gtl::InlinedVector<int, 4> indices(num_dims);
    int offset = 0;  // the shift along the flattened tensor for current element
    // initialize indices and offset
    for (int i = 0; i < num_dims; i++) {
      // stride is the number of indices over in the flattened tensor
      // you need to skip in order to make it over to an adjacent element
      // along a dimension. dim_size[i] != 0 because we set it to max(dim, 1)
      const int64 stride = dim_range[i] / dim_size[i];
      const int shift = dim_size[i] - threshold[i];
      const int indx = (start / stride) % dim_size[i];
      indices[i] = indx;
      // calculate dimension index after the shift
      const int shifted_indx = (indx + shift) % dim_size[i];
      offset += (shifted_indx - indx) * stride;
    }

    for (int64 i = start; i < end; i++) {
      output[i + offset] = input[i];
      // create next combination of indices
      // while at it adjust offset if needed
      for (int j = num_dims - 1; j >= 0; j--) {
        const int indx = (indices[j] + 1) % dim_size[j];
        indices[j] = indx;
        if (indx != 0) {
          if (indx == threshold[j]) {  // we've reached the threshold
            // dim_range[j] = threshold[j] + shift[j]
            // offset = shift[j] + ... other offsets
            // offset - dim_range[j] = -threshold[j] + ... other offsets
            // thus we undo our previous offset as well as add a new offset of
            // -threshold[j] in one operation
            offset -= dim_range[j];  // now wraps around
          }
          break;                         // indx != 0 don't need to carry
        } else if (threshold[j] != 0) {  // if threshold is 0 shift is 0
          offset += dim_range[j];        // indx became 0 so reverse wrap around
        }
      }
    }
  };
  // Shard
  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  // 15 - expiramentally determined with float and bool types
  const int cost_per_element = 15 * sizeof(T);  // rough estimate
  Shard(worker_threads->num_threads, worker_threads->workers, num_elements,
        cost_per_element, std::move(work));
}

// dim_size - the size of each dimension
// dim_range - the number of indices over in the flattened tensor
//    you need to skip in order to make it over from one side of a dimension
//    to the other. Used to make the shifts wrap around after a threshold.
// threshold - the index for each dimension that the roll starts to wrap
//    back to the front
// isd - inner shift dimension
template <typename T>
// Use memcpy to copy memory in groups when the data type supports memcpy
void DoRollWithMemcpy(OpKernelContext* context, const int64 num_elements,
                      const int num_dims, const gtl::ArraySlice<int>& dim_size,
                      const T* input, T* output,
                      const gtl::ArraySlice<int>& threshold,
                      const gtl::ArraySlice<int64>& dim_range,
                      const int64 isd) {
  auto work = [input, output, num_dims, &dim_size, &threshold, &dim_range, isd](
                  int64 start, int64 end) {
    // the number of indices over in the flattened tensor you need to skip in
    // order to make it over from one side of the isd to the other
    const int64 isd_range = std::max<int>(dim_range[isd], 1);
    // the distance along the flattend tensor to the next element in the isd
    const int64 isd_stride = isd_range / std::max<int>(dim_size[isd], 1);

    // start and end represent the i-th group currently so we will convert
    // them into numbers representing the i-th elements.
    // there are 2 groups per isd one for all elements before threshold[isd]
    // and another for all elements after threshold[isd].
    const int64 start_remainder = (start % 2) * threshold[isd] * isd_stride;
    const int64 end_remainder = (end % 2) * threshold[isd] * isd_stride;
    start = (start / 2) * isd_range + start_remainder;
    end = (end / 2) * isd_range + end_remainder;

    const T* in_ptr = &input[0];
    T* out_ptr = &output[0];
    in_ptr += start;
    out_ptr += start;

    // array of indices for each dimension
    // indicies = [i, j, k, l, m, n]
    gtl::InlinedVector<int, 4> indicies(num_dims);
    // the offset needed to make all inner non-shifting dimensions become 0
    int64 remainder_offset = 0;
    // initialize indicies
    for (int i = 0; i < num_dims; i++) {
      // stride is the number of indices over in the flattened tensor
      // you need to skip in order to make it over to an adjacent element
      // along a dimension. dim_size[i] != 0 because we set it to max(dim, 1)
      const int64 stride = dim_range[i] / dim_size[i];
      const int shift = dim_size[i] - threshold[i];
      const int indx = (start / stride) % dim_size[i];
      indicies[i] = indx;
      // calculate dimension index after the shift
      int out_indx = (indx + shift) % dim_size[i];
      if (i > isd) {
        // trailing zeroes for indices after the inner shifted dimension
        out_indx = 0;
        remainder_offset += (out_indx - indx) * stride;
      }
      out_ptr += (out_indx - indx) * stride;
    }
    // set trailing zeroes for indices after the inner shifted dimension
    for (int i = num_dims - 1; i > isd; i--) indicies[i] = 0;

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
      // the isd (inner shift dim) should skip to next threshold or endpoint
      // all dimensions to the left increment by 1 when a digit is carried
      // all dimensions to the right remain set to 0
      //            +1 +1 +1 +isd_indx_skip
      // indicies = [i, j, k, l, 0, 0]
      //                      ^isd
      for (int j = isd; j >= 0; j--) {
        int inc = 1;
        if (j == isd) inc = isd_indx_skip;
        const int indx = (indicies[j] + inc) % dim_size[j];
        indicies[j] = indx;
        if (indx != 0) {
          if (indx == threshold[j]) {
            out_ptr -= dim_range[j];  // now wraps around
          }
          break;                         // indx != 0 don't need to carry
        } else if (threshold[j] != 0) {  // if threshold is 0 shift is 0
          out_ptr += dim_range[j];       // indx became 0 so reverse wrap around
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
  const int total_work = 2 * num_elements / std::max<int>(dim_range[isd], 1);
  // 25000 - expiramentally determined with float and bool types
  const int cost_per_group = 25000 * sizeof(T) * ave_group_size;
  Shard(worker_threads->num_threads, worker_threads->workers, total_work,
        cost_per_group, std::move(work));
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
        errors::InvalidArgument("shift and axis must have the same size"));
    const int64 num_elements = input.NumElements();
    const int num_shifts = static_cast<int>(shift_flat.size());
    const int num_dims = input.dims();

    // if there are any duplicate axes, shift_mod_sum will have the
    // total modulo sum of shifts for each dimension
    gtl::InlinedVector<int, 4> shift_mod_sum(num_dims, 0);
    for (int i = 0; i < num_shifts; i++) {
      int axis = axis_flat(i);
      if (axis < 0) {
        axis += num_dims;
      }
      OP_REQUIRES(context, 0 <= axis && axis < num_dims,
                  errors::InvalidArgument("axis ", axis, " is out of range"));
      const int ds = std::max<int>(static_cast<int>(input.dim_size(axis)), 1);
      const int sum = shift_mod_sum[axis] + static_cast<int>(shift_flat(i));
      // modulo that works with negatives: ((x % y) + y) % y
      shift_mod_sum[axis] = (sum % ds + ds) % ds;
    }
    // the size of each dimension
    gtl::InlinedVector<int, 4> dim_size(num_dims);
    // threshold[i] is the index that the roll starts to wrap back to the front
    gtl::InlinedVector<int, 4> threshold(num_dims);
    // dim_range is the number of indices over in the flattened tensor
    // you need to skip in order to make it over from one side of a dimension
    // to the other. Used to make the shifts wrap around after a threshold.
    gtl::InlinedVector<int64, 4> dim_range(num_dims);
    int64 dim_size_prod = 1;  // dimension size product
    // inner shift dimension (inner most shifted dimension)
    int64 isd = 0;
    for (int i = num_dims - 1; i >= 0; i--) {
      if (isd == 0 && shift_mod_sum[i] != 0) isd = i;
      const int ds = std::max<int>(static_cast<int>(input.dim_size(i)), 1);
      dim_size[i] = ds;
      threshold[i] = (ds - shift_mod_sum[i]) % ds;
      dim_size_prod *= static_cast<int64>(input.dim_size(i));
      dim_range[i] = dim_size_prod;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    auto input_flat = input.flat<T>().data();
    auto output_flat = output->flat<T>().data();

    if (std::is_same<Device, CPUDevice>::value) {
      if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
        // V2 copies memory in groups instead of element by element
        DoRollWithMemcpy<T>(context, num_elements, num_dims, dim_size,
                            input_flat, output_flat, threshold, dim_range, isd);
      } else {
        // incase memcpy does not work for current data type
        DoRoll<T>(context, num_elements, num_dims, dim_size, input_flat,
                  output_flat, threshold, dim_range);
      }
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
#undef REGISTER_CPU
}  // namespace tensorflow
