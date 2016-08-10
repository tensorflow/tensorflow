/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/strided_slice_op.h"
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/strided_slice_op_impl.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

namespace {

/// Constants
constexpr int32 kShrinkAxis = -1, kNewAxis = -2;

struct StridedSliceSparseSpec {
  int64 dims;
  int32 num_add_axis_after_ellipsis;
  const Tensor& begin_tensor;
  const Tensor& end_tensor;
  const Tensor& strides_tensor;
  const int32 begin_mask, end_mask;
  int32 ellipsis_mask;
  const int32 new_axis_mask, shrink_axis_mask;
};

struct StridedSliceDenseSpec {
  const int64 dims;
  int32 begin_mask;
  int32 end_mask;
  gtl::InlinedVector<int64, 4>& begin;
  gtl::InlinedVector<int64, 4>& end;
  gtl::InlinedVector<int64, 4>& strides;
  gtl::InlinedVector<int32, 4> final_shape_gather_indices;
};

}  // namespace

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <class T>
static void BuildDenseSpec(const StridedSliceSparseSpec& sparse,
                           StridedSliceDenseSpec* dense) {
  // Build expanded begin, end, strides, begin_mask, end_mask
  // to remove any ellipsis
  dense->begin.resize(dense->dims);
  dense->end.resize(dense->dims);
  dense->strides.resize(dense->dims);
  // What indices to get the final shape from.
  dense->begin_mask = 0;
  dense->end_mask = 0;
  {
    int full_index = 0;

    const auto& begin_flat = sparse.begin_tensor.flat<T>();
    const auto& end_flat = sparse.end_tensor.flat<T>();
    const auto& strides_flat = sparse.strides_tensor.flat<T>();

    for (int i = 0; i < sparse.dims; i++) {
      if ((1 << i) & sparse.ellipsis_mask) {
        // Expand the ellipsis into the appropriate indices
        // NOTE: this only works because we guaranteed one ellipsis
        int32 next_index = std::min(dense->dims - (sparse.dims - i) + 1 +
                                        sparse.num_add_axis_after_ellipsis,
                                    dense->dims);
        for (; full_index < next_index; full_index++) {
          // new_axis' aren't real axis so you have to skip
          dense->begin[full_index] = dense->end[full_index] = 0;
          dense->strides[full_index] = 1;
          dense->begin_mask |= (1 << full_index);
          dense->end_mask |= (1 << full_index);
          dense->final_shape_gather_indices.push_back(full_index);
        }
      } else if ((1 << i) & sparse.new_axis_mask) {
        dense->final_shape_gather_indices.push_back(kNewAxis);
      } else {
        // Gather slicing spec into appropriate index
        dense->begin[full_index] = internal::SubtleMustCopy<T>(begin_flat(i));
        dense->end[full_index] = internal::SubtleMustCopy<T>(end_flat(i));
        dense->strides[full_index] =
            internal::SubtleMustCopy<T>(strides_flat(i));
        if (sparse.begin_mask & (1 << i)) {
          dense->begin_mask |= (1 << full_index);
        }
        if (sparse.end_mask & (1 << i)) {
          dense->end_mask |= (1 << full_index);
        }
        // If shrink, use the shrink code, otherwise use the real value
        dense->final_shape_gather_indices.push_back(
            (sparse.shrink_axis_mask & (1 << i)) ? kShrinkAxis : full_index);
        full_index++;
      }
    }
  }
}

// Shared code that is not dependent on the type of T.  We do this to reduce
// code size by not duplicating all this for all T (float, double, int32, etc.)
static void SharedValidation(
    OpKernelContext* context, const TensorShape& input_shape,
    int32 begin_mask_spec, int32 end_mask_spec, const int32 ellipsis_mask,
    int32 new_axis_mask, int32 shrink_axis_mask, TensorShape* processing_shape,
    TensorShape* final_shape, bool* is_identity, bool* is_simple_slice,
    bool* slice_dim0, gtl::InlinedVector<int64, 4>* begin,
    gtl::InlinedVector<int64, 4>* end, gtl::InlinedVector<int64, 4>* strides) {
  const Tensor& begin_tensor = context->input(1);
  const Tensor& end_tensor = context->input(2);
  const Tensor& strides_tensor = context->input(3);
  OP_REQUIRES(
      context, TensorShapeUtils::IsVector(begin_tensor.shape()) &&
                   TensorShapeUtils::IsVector(end_tensor.shape()) &&
                   TensorShapeUtils::IsVector(strides_tensor.shape()) &&
                   strides_tensor.dims() == 1 &&
                   strides_tensor.dims() == begin_tensor.dims() &&
                   strides_tensor.dims() == end_tensor.dims() &&
                   begin_tensor.dim_size(0) == end_tensor.dim_size(0) &&
                   begin_tensor.dim_size(0) == strides_tensor.dim_size(0) &&
                   begin_tensor.dim_size(0) < 32,  // using 32 bit masks
      errors::InvalidArgument(
          "Expected begin, end, and strides to be 1D equal size tensors, ",
          "but got shapes ", begin_tensor.shape().DebugString(), ", ",
          end_tensor.shape().DebugString(), ", and ",
          strides_tensor.shape().DebugString(), " instead."));
  // Use bit compares to ensure ellipsis_mask is 0 or a power of 2
  // i.e. there exists only no more than one ellipsis
  OP_REQUIRES(context,
              !ellipsis_mask || (ellipsis_mask & (ellipsis_mask - 1)) == 0,
              errors::InvalidArgument("Multiple ellipsis' in slice "
                                      "spec not allowed"));

  // Step 1: Account for ellipsis and new axis
  //
  // Check for ellipses and count how many non-newaxis' there are after
  // TODO(aselle): Convert this to do a fast log2 followed by iteration
  //               counting ones in next guys
  bool ellipsis_seen = false;

  StridedSliceSparseSpec sparse_spec = {begin_tensor.NumElements(),
                                        0,
                                        begin_tensor,
                                        end_tensor,
                                        strides_tensor,
                                        begin_mask_spec,
                                        end_mask_spec,
                                        ellipsis_mask,
                                        new_axis_mask,
                                        shrink_axis_mask};

  for (int32 i = 0; i < sparse_spec.dims; i++) {
    if (ellipsis_seen && ((1 << i) & new_axis_mask) != 0) {
      sparse_spec.num_add_axis_after_ellipsis++;
    }
    if ((1 << i) & ellipsis_mask) {
      ellipsis_seen = true;
    }
  }
  // If no ellipsis insert one at the end
  if (!ellipsis_seen) {
    sparse_spec.ellipsis_mask |= (1 << sparse_spec.dims);
    sparse_spec.dims++;  // this effects loop iteration below
  }

  // Step 2: Make a sparse spec into a full index spec
  //
  // The sparse spec does not corresopnds to the number of dimensions
  // Make a dense spec that corresponds to thte number of dimensions
  //
  // For example suppose foo[...,3:] on foo.shape=(2,2,3) then
  // we need to produce the missing begin_mask for the the first two
  // dimensions i.e. from begin_mask_spec=0, end_mask_spec=2
  // we achieve begin_mask=6, end_mask=7
  StridedSliceDenseSpec dense_spec = {
      input_shape.dims(), 0, 0, *begin, *end, *strides};

  if (begin_tensor.dtype() == DT_INT32) {
    BuildDenseSpec<int32>(sparse_spec, &dense_spec);
  } else if (begin_tensor.dtype() == DT_INT64) {
    BuildDenseSpec<int64>(sparse_spec, &dense_spec);
  } else {
    LOG(FATAL) << "begin must be either int32 or int64";
  }

  // Step 3: Make implicit ranges (non-zero begin_masks and end_masks) explicit
  //         and bounds check!
  *is_identity = true;
  *slice_dim0 = true;
  *is_simple_slice = true;
  for (int i = 0; i < dense_spec.dims; ++i) {
    int64& begin_i = (*begin)[i];
    int64& end_i = (*end)[i];
    int64& stride_i = (*strides)[i];
    int64 dim_i = input_shape.dim_size(i);
    OP_REQUIRES(context, stride_i != 0,
                errors::InvalidArgument("strides[", i, "] must be non-zero"));

    int64 masks[] = {dense_spec.begin_mask & (1 << i),
                     dense_spec.end_mask & (1 << i)};
    int64 valid_range[] = {stride_i > 0 ? 0 : -1,
                           stride_i > 0 ? dim_i : dim_i - 1};

    auto canonical = [stride_i, i, dim_i, masks, valid_range](int64 x, int c) {
      if (masks[c]) {
        return stride_i > 0 ? valid_range[c] : valid_range[(c + 1) & 1];
      } else {
        int64 x_fwd = x < 0 ? dim_i + x : x;  // make negative indices positive
        return x_fwd < valid_range[0]
                   ? valid_range[0]
                   : x_fwd > valid_range[1] ? valid_range[1] : x_fwd;
      }
    };
    begin_i = canonical(begin_i, 0);
    end_i = canonical(end_i, 1);
    // Update optimization values
    (*is_simple_slice) &= stride_i == 1;
    bool take_all_in_dimension =
        stride_i == 1 && begin_i == 0 && end_i == input_shape.dim_size(i);
    (*is_identity) &= take_all_in_dimension;
    (*slice_dim0) &= (i == 0 && stride_i == 1) || take_all_in_dimension;

    // Compute the processing shape (the intermediate Eigen will produce)
    int64 interval_length = end_i - begin_i;
    int64 size_i;
    // Hold zero if the interval is degenerate, otherwise account for remainder
    if (interval_length == 0 || ((interval_length < 0) != (stride_i < 0)))
      size_i = 0;
    else
      size_i = interval_length / stride_i +
               (interval_length % stride_i != 0 ? 1 : 0);
    processing_shape->AddDim(size_i);
  }

  // Step 4: Compute the final shape
  //
  // new_axis will increase dimension by 1 (with a one-size dimension)
  // slices like foo[3,...] will reduce dimension by 1.
  // This cannot be done earlier, because it depends on Step 3.
  for (auto gather_index : dense_spec.final_shape_gather_indices) {
    if (gather_index >= 0)
      final_shape->AddDim(processing_shape->dim_size(gather_index));
    else if (gather_index == kNewAxis)
      final_shape->AddDim(1);
  }
}

template <typename Device, typename T>
class StridedSliceOp : public OpKernel {
 public:
  explicit StridedSliceOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
  }

  void Compute(OpKernelContext* context) override {
    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    SharedValidation(context, context->input(0).shape(), begin_mask, end_mask,
                     ellipsis_mask, new_axis_mask, shrink_axis_mask,
                     &processing_shape, &final_shape, &is_identity,
                     &is_simple_slice, &slice_dim0, &begin, &end, &strides);
    if (!context->status().ok()) return;

    const Tensor& input = context->input(0);

    // Optimization #1, slice is a no-op plus reshape
    if (is_identity) {
      Tensor tmp;
      CHECK(tmp.CopyFrom(input, final_shape));
      context->set_output(0, tmp);
      return;
    }

    // Optimization #2, slice is memory contiguous (only occurs in dim 0)
    if (slice_dim0 && IsInnerDimsSizeAligned<T>(input.shape())) {
      CHECK_GE(input.dims(), 1);  // Otherwise, is_identity should be true.
      Tensor tmp;
      CHECK(tmp.CopyFrom(input.Slice(begin[0], end[0]), final_shape));
      context->set_output(0, tmp);
      return;
    }

    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, final_shape, &result));
    const int input_dims = input.dims();
    const int processing_dims = processing_shape.dims();

    if (processing_shape.num_elements() > 0) {
      // Optimization #3, slice has stride 1 in all dimensions
      // Optimization #3A, slice has only two dimensions
      // TODO(aselle): Here we are restricting to processing_shape and
      // final_shape being 2D. This isn't strictly necessary, but I don't
      // want to blow up code gen size, because to shape<> you need static
      // NDIM and T
      if (is_simple_slice && std::is_same<Device, CPUDevice>::value &&
          input_dims == 2 && processing_shape.dims() == 2 &&
          final_shape.dims() == 2 &&
          DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
        auto in = input.tensor<T, 2>();
        auto output = result->tensor<T, 2>();
        // TODO(agarwal): Consider multi-threading if size[0] is large
        for (int row_in = begin[0], row_out = 0; row_in < end[0];
             ++row_in, ++row_out) {
          if (row_in + 1 < end[0]) {
            port::prefetch<port::PREFETCH_HINT_T0>(&output(row_in + 1, 0));
            port::prefetch<port::PREFETCH_HINT_T0>(&in(row_in + 1, begin[1]));
          }
          memcpy(&output(row_out, 0), &in(row_in, begin[1]),
                 (end[1] - begin[1]) * sizeof(T));
        }
        return;
      }

#define HANDLE_DIM(NDIM)                                                       \
  if (processing_dims == NDIM) {                                               \
    HandleStridedSliceCase<Device, T, NDIM>(context, begin, end, strides,      \
                                            processing_shape, is_simple_slice, \
                                            result);                           \
    return;                                                                    \
  }

      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);

#undef HANDLE_DIM

      OP_REQUIRES(
          context, false,
          errors::Unimplemented("Unhandled input dimensions ", input_dims));
    }
  }

 private:
  int32 begin_mask, end_mask;
  int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

template <typename Device, typename T>
class StridedSliceGradOp : public OpKernel {
 public:
  explicit StridedSliceGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
  }

  void Compute(OpKernelContext* context) override {
    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64, 4> begin;
    gtl::InlinedVector<int64, 4> end;
    gtl::InlinedVector<int64, 4> strides;

    TensorShape input_shape;
    const Tensor& input_shape_tensor = context->input(0);
    OP_REQUIRES(
        context, input_shape_tensor.dims() == 1,
        errors::InvalidArgument("shape must be 1-D, got shape.shape = ",
                                input_shape_tensor.shape().DebugString()));
    if (input_shape_tensor.dtype() == DT_INT32) {
      OP_REQUIRES_OK(
          context, TensorShapeUtils::MakeShape(input_shape_tensor.vec<int32>(),
                                               &input_shape));
    } else if (input_shape_tensor.dtype() == DT_INT64) {
      OP_REQUIRES_OK(
          context, TensorShapeUtils::MakeShape(input_shape_tensor.vec<int64>(),
                                               &input_shape));
    } else {
      LOG(FATAL) << "shape must have type int32 or int64.";
    }

    SharedValidation(context, input_shape, begin_mask, end_mask, ellipsis_mask,
                     new_axis_mask, shrink_axis_mask, &processing_shape,
                     &final_shape, &is_identity, &is_simple_slice, &slice_dim0,
                     &begin, &end, &strides);

    // Check to make sure dy is consistent with the original slice
    TensorShape dy_shape = context->input(4).shape();
    OP_REQUIRES(
        context, final_shape == dy_shape,
        errors::InvalidArgument("shape of dy was ", dy_shape.DebugString(),
                                " instead of ", final_shape.DebugString()));

    if (!context->status().ok()) return;

    // const int input_dims = input.dims();
    const int processing_dims = processing_shape.dims();
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &result));

#define HANDLE_DIM(NDIM)                                                      \
  if (processing_dims == NDIM) {                                              \
    HandleStridedSliceGradCase<Device, T, NDIM>(context, begin, end, strides, \
                                                processing_shape,             \
                                                is_simple_slice, result);     \
    return;                                                                   \
  }

    HANDLE_DIM(1);
    HANDLE_DIM(2);
    HANDLE_DIM(3);
    HANDLE_DIM(4);
    HANDLE_DIM(5);
    HANDLE_DIM(6);

#undef HANDLE_DIM
  }

 private:
  int32 begin_mask, end_mask;
  int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

#define REGISTER_STRIDED_SLICE(type)                       \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")             \
                              .Device(DEVICE_CPU)          \
                              .TypeConstraint<type>("T")   \
                              .HostMemory("begin")         \
                              .HostMemory("end")           \
                              .HostMemory("strides"),      \
                          StridedSliceOp<CPUDevice, type>) \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")         \
                              .Device(DEVICE_CPU)          \
                              .TypeConstraint<type>("T")   \
                              .HostMemory("shape")         \
                              .HostMemory("begin")         \
                              .HostMemory("end")           \
                              .HostMemory("strides"),      \
                          StridedSliceGradOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_STRIDED_SLICE);
REGISTER_STRIDED_SLICE(bfloat16);

#undef REGISTER_STRIDED_SLICE

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                     \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")                 \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("T")       \
                              .HostMemory("begin")             \
                              .HostMemory("end")               \
                              .HostMemory("strides")           \
                              .TypeConstraint<int32>("Index"), \
                          StridedSliceOp<GPUDevice, type>)     \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")             \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<type>("T")       \
                              .HostMemory("shape")             \
                              .HostMemory("begin")             \
                              .HostMemory("end")               \
                              .HostMemory("strides")           \
                              .TypeConstraint<int32>("Index"), \
                          StridedSliceGradOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("StridedSlice")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Index")
                            .HostMemory("input")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("output"),
                        StridedSliceOp<CPUDevice, int32>);

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
