/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/ragged_fill_empty_rows_op.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T, typename Tindex>
struct RaggedFillEmptyRows<CPUDevice, T, Tindex> {
 public:
  Status operator()(OpKernelContext* context, const Tensor& default_value_t,
                    const Tensor& value_rowids_t, const Tensor& values_t,
                    const Tensor& nrows_t,
                    typename AsyncOpKernel::DoneCallback done) {
    (void)done;  // Unused (only used in GPU implementation)
    const int kOutputValueRowidsOutput = 0;
    const int kOutputValuesOutput = 1;
    const int kEmptyRowIndicatorOutput = 2;
    const int kReverseIndexMapOutput = 3;

    const T& default_value = default_value_t.scalar<T>()();
    const auto value_rowids = value_rowids_t.vec<Tindex>();
    const auto values = values_t.vec<T>();
    const Tindex& nrows = nrows_t.scalar<Tindex>()();

    const Tindex N = value_rowids_t.shape().dim_size(0);

    bool* empty_row_indicator = nullptr;
    if (context->output_required(kEmptyRowIndicatorOutput)) {
      Tensor* empty_row_indicator_t = nullptr;
      TensorShape output_shape;
      TF_RETURN_IF_ERROR(
          TensorShape::BuildTensorShape({nrows}, &output_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kEmptyRowIndicatorOutput, output_shape, &empty_row_indicator_t));
      empty_row_indicator = empty_row_indicator_t->vec<bool>().data();
    }
    Tindex* reverse_index_map = nullptr;
    if (context->output_required(kReverseIndexMapOutput)) {
      Tensor* reverse_index_map_t = nullptr;
      TensorShape output_shape;
      TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape({N}, &output_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kReverseIndexMapOutput, output_shape, &reverse_index_map_t));
      reverse_index_map = reverse_index_map_t->vec<Tindex>().data();
    }

    if (nrows == 0) {
      if (N != 0) {
        return errors::InvalidArgument(
            "Received RaggedTensor with nrows = 0 but "
            "va.shape[0] = ",
            N);
      }
      Tensor* output_value_rowids_t;
      TensorShape output_value_rowids_shape;
      TF_RETURN_IF_ERROR(
          TensorShape::BuildTensorShape({0}, &output_value_rowids_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputValueRowidsOutput, output_value_rowids_shape, &output_value_rowids_t));
      Tensor* output_values_t;
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputValuesOutput, TensorShape({0}), &output_values_t));

      // Exit early, nothing more to do.
      return OkStatus();
    }

    bool rows_are_ordered = true;
    Tindex last_value_rowids_row = 0;
    std::vector<Tindex> csr_offset(nrows, 0);
    for (int i = 0; i < N; ++i) {
      const Tindex row = value_rowids(i);
      if (row < 0 || row >= nrows) {
        return errors::InvalidArgument("value_rowids(", i, ", 0) is invalid: ", row,
                                       " >= ", nrows);
      }
      ++csr_offset[row];
      rows_are_ordered = rows_are_ordered & (row >= last_value_rowids_row);
      last_value_rowids_row = row;
    }
    bool all_rows_full = true;
    for (int row = 0; row < nrows; ++row) {
      // csr_offset here describes the number of elements in this dense row
      bool row_empty = (csr_offset[row] == 0);
      if (empty_row_indicator) {
        empty_row_indicator[row] = row_empty;
      }
      all_rows_full = all_rows_full & !row_empty;
      // In filled version, each row has at least one element.
      csr_offset[row] = std::max(csr_offset[row], Tindex{1});
      // Update csr_offset to represent the number of elements up to and
      // including dense_row + 1:
      //  csr_offset(0) == #{elements of row 0}
      //  csr_offset(1) == #{elements of row 1} + #{elements of row 0}
      //  ..
      //  csr_offset(i) == starting index for elements in row i + 1.
      if (row > 0) {
        csr_offset[row] += csr_offset[row - 1];
      }
    }

    if (all_rows_full && rows_are_ordered) {
      context->set_output(kOutputValueRowidsOutput, value_rowids_t);
      context->set_output(kOutputValuesOutput, values_t);
      if (reverse_index_map) {
        for (Tindex i = 0; i < N; ++i) {
          reverse_index_map[i] = i;
        }
      }
    } else {
      Tensor* output_value_rowids_t;
      const Tindex N_full = csr_offset[nrows - 1];
      TensorShape output_value_rowids_shape;
      TF_RETURN_IF_ERROR(
          TensorShape::BuildTensorShape({N_full}, &output_value_rowids_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputValueRowidsOutput, output_value_rowids_shape, &output_value_rowids_t));
      auto output_value_rowids = output_value_rowids_t->vec<Tindex>();

      Tensor* output_values_t;
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputValuesOutput, TensorShape({N_full}), &output_values_t));
      auto output_values = output_values_t->vec<T>();

      std::vector<Tindex> filled_count(nrows, 0);

      // Fill in values for rows that are not missing
      for (Tindex i = 0; i < N; ++i) {
        const Tindex row = value_rowids(i);
        Tindex& offset = filled_count[row];
        const Tindex output_i = ((row == 0) ? 0 : csr_offset[row - 1]) + offset;
        offset++;  // Increment the filled count for this row.
        std::copy_n(&value_rowids(i), 1, &output_value_rowids(output_i));
        output_values(output_i) = values(i);
        // We'll need this reverse index map to backprop correctly.
        if (reverse_index_map) {
          reverse_index_map[i] = output_i;
        }
      }

      // Fill in values for rows that are missing
      for (Tindex row = 0; row < nrows; ++row) {
        const Tindex row_count = filled_count[row];
        if (row_count == 0) {  // We haven't filled this row
          const Tindex starting_index = (row == 0) ? 0 : csr_offset[row - 1];
          // Remaining index values were set to zero already.
          // Just need to set the row index in the right location.
          output_value_rowids(starting_index) = row;
          output_values(starting_index) = default_value;
        }
      }
    }

    return OkStatus();
  }
};

}  // namespace functor

namespace {

template <typename Device, typename T, typename Tindex>
void RaggedFillEmptyRowsOpImpl(OpKernelContext* context,
                               AsyncOpKernel::DoneCallback done = nullptr) {
  // Note that setting this empty lambda as the default parameter value directly
  // can cause strange compiler/linker errors, so we do it like this instead.
  if (!done) {
    done = [] {};
  }

  const int kValueRowidsInput = 0;
  const int kValuesInput = 1;
  const int kNRowsInput = 2;
  const int kDefaultValueInput = 3;

  const Tensor& value_rowids_t = context->input(kValueRowidsInput);
  const Tensor& values_t = context->input(kValuesInput);
  const Tensor& nrows_t = context->input(kNRowsInput);
  const Tensor& default_value_t = context->input(kDefaultValueInput);

  OP_REQUIRES_ASYNC(
      context, TensorShapeUtils::IsScalar(nrows_t.shape()),
      errors::InvalidArgument("nrows must be a scalar, saw: ",
                              nrows_t.shape().DebugString()),
      done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(value_rowids_t.shape()),
                  errors::InvalidArgument("value_rowids must be a vector, saw: ",
                                            value_rowids_t.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(values_t.shape()),
                    errors::InvalidArgument("values must be a vector, saw: ",
                                            values_t.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(
      context, value_rowids_t.dim_size(0) == values_t.dim_size(0),
      errors::InvalidArgument("The length of `values` (", values_t.dim_size(0),
                              ") must match the first dimension of `value_rowids` (",
                              value_rowids_t.dim_size(0), ")."),
      done);
  OP_REQUIRES_ASYNC(
      context, TensorShapeUtils::IsScalar(default_value_t.shape()),
      errors::InvalidArgument("default_value must be a scalar, saw: ",
                              default_value_t.shape().DebugString()),
      done);

  using FunctorType =
      functor::RaggedFillEmptyRows<Device, T, Tindex>;
  OP_REQUIRES_OK_ASYNC(context,
                       FunctorType()(context, default_value_t, value_rowids_t,
                                     values_t, nrows_t, done),
                       done);
}

}  // namespace

template <typename Device, typename T, typename Tindex>
class RaggedFillEmptyRowsOp : public OpKernel {
 public:
  explicit RaggedFillEmptyRowsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    RaggedFillEmptyRowsOpImpl<Device, T, Tindex>(context);
  }

};

#define REGISTER_KERNELS(D, T, Tindex)                   \
  REGISTER_KERNEL_BUILDER(Name("RaggedFillEmptyRows")    \
                              .Device(DEVICE_##D)        \
                              .HostMemory("nrows") \
                              .TypeConstraint<T>("T"),   \
                          RaggedFillEmptyRowsOp<D##Device, T, Tindex>)

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T, int64)
TF_CALL_ALL_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// The GPU implementation is async because it requires waiting for a
// host->device memcpy before the output is allocated (similar to
// SegmentSumGPUOp).
template <typename T, typename Tindex>
class RaggedFillEmptyRowsGPUOp : public AsyncOpKernel {
 public:
  explicit RaggedFillEmptyRowsGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    RaggedFillEmptyRowsOpImpl<GPUDevice, T, Tindex>(context, done);
  }
};

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                            \
  template <>                                                                  \
  Status RaggedFillEmptyRows<GPUDevice, T, Tindex>::operator()(          \
      OpKernelContext* context, const Tensor& default_value_t,                 \
      const Tensor& value_rowids_t, const Tensor& values_t,                         \
      const Tensor& nrows_t, typename AsyncOpKernel::DoneCallback done); \
  extern template struct RaggedFillEmptyRows<GPUDevice, T, Tindex>;
#define DECLARE_GPU_SPEC_INT64(T) DECLARE_GPU_SPEC(T, int64_t)
TF_CALL_POD_TYPES(DECLARE_GPU_SPEC_INT64)
#undef DECLARE_GPU_SPEC_INT64
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_KERNELS(T, Tindex)                      \
  REGISTER_KERNEL_BUILDER(Name("RaggedFillEmptyRows")    \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("nrows") \
                              .TypeConstraint<T>("T"),   \
                          RaggedFillEmptyRowsGPUOp<T, Tindex>)

#define REGISTER_KERNELS_TINDEX(T) REGISTER_KERNELS(T, int64)
TF_CALL_POD_TYPES(REGISTER_KERNELS_TINDEX)
#undef REGISTER_KERNELS_TINDEX

#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

template <typename T, typename Tindex>
struct RaggedFillEmptyRowsGrad<CPUDevice, T, Tindex> {
  Status operator()(OpKernelContext* context,
                    typename TTypes<Tindex>::ConstVec reverse_index_map,
                    typename TTypes<T>::ConstVec grad_values,
                    typename TTypes<T>::Vec d_values,
                    typename TTypes<T>::Scalar d_default_value) {
    const CPUDevice& device = context->eigen_device<CPUDevice>();
    const Tindex N = reverse_index_map.dimension(0);
    const Tindex N_full = grad_values.dimension(0);

    T& d_default_value_scalar = d_default_value();
    d_default_value_scalar = T();

    Tensor visited_t;
    TF_RETURN_IF_ERROR(
        context->allocate_temp(DT_BOOL, TensorShape({N_full}), &visited_t));
    auto visited = visited_t.vec<bool>();
    visited.device(device) = visited.constant(false);

    for (int i = 0; i < N; ++i) {
      // Locate the index of the output of the forward prop associated
      // with this location in the input of the forward prop.  Copy
      // the gradient into it.  Mark it as visited.
      int64_t reverse_index = reverse_index_map(i);
      if (reverse_index < 0 || reverse_index >= N_full) {
        return errors::InvalidArgument(
            "Elements in reverse index must be in [0, ", N_full, ") but got ",
            reverse_index);
      }
      d_values(i) = grad_values(reverse_index);
      visited(reverse_index) = true;
    }
    for (int j = 0; j < N_full; ++j) {
      // The default value gradient gets the accumulated remainder of
      // the backprop values (since the default value was used to fill
      // in these slots in the forward calculation).
      if (!visited(j)) {
        d_default_value_scalar += grad_values(j);
      }
    }
    return OkStatus();
  }
};

}  // namespace functor

template <typename Device, typename T, typename Tindex>
class RaggedFillEmptyRowsGradOp : public OpKernel {
 public:
  explicit RaggedFillEmptyRowsGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* reverse_index_map_t;
    const Tensor* grad_values_t;
    OP_REQUIRES_OK(context,
                   context->input("reverse_index_map", &reverse_index_map_t));
    OP_REQUIRES_OK(context, context->input("grad_values", &grad_values_t));

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(reverse_index_map_t->shape()),
        errors::InvalidArgument("reverse_index_map must be a vector, saw: ",
                                reverse_index_map_t->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(grad_values_t->shape()),
                errors::InvalidArgument("grad_values must be a vector, saw: ",
                                        grad_values_t->shape().DebugString()));

    const auto reverse_index_map = reverse_index_map_t->vec<Tindex>();
    const auto grad_values = grad_values_t->vec<T>();

    const Tindex N = reverse_index_map_t->shape().dim_size(0);

    Tensor* d_values_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "d_values", TensorShape({N}), &d_values_t));
    auto d_values = d_values_t->vec<T>();
    Tensor* d_default_value_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output("d_default_value", TensorShape({}),
                                            &d_default_value_t));
    auto d_default_value = d_default_value_t->scalar<T>();

    OP_REQUIRES_OK(context,
                   functor::RaggedFillEmptyRowsGrad<Device, T, Tindex>()(
                       context, reverse_index_map, grad_values, d_values,
                       d_default_value));
  }
};

#define REGISTER_KERNELS(D, T, Tindex)                    \
  REGISTER_KERNEL_BUILDER(Name("RaggedFillEmptyRowsGrad") \
                              .Device(DEVICE_##D)         \
                              .TypeConstraint<T>("T"),    \
                          RaggedFillEmptyRowsGradOp<D##Device, T, Tindex>)

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T, int64)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                 \
  template <>                                                       \
  Status RaggedFillEmptyRowsGrad<GPUDevice, T, Tindex>::operator()( \
      OpKernelContext* context,                                     \
      typename TTypes<Tindex>::ConstVec reverse_index_map,          \
      typename TTypes<T>::ConstVec grad_values,                     \
      typename TTypes<T>::Vec d_values,                             \
      typename TTypes<T>::Scalar d_default_value);                  \
  extern template struct RaggedFillEmptyRowsGrad<GPUDevice, T, Tindex>;
#define DECLARE_GPU_SPEC_INT64(T) DECLARE_GPU_SPEC(T, int64_t)
TF_CALL_REAL_NUMBER_TYPES(DECLARE_GPU_SPEC_INT64);
#undef DECLARE_GPU_SPEC_INT64
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T, int64)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_KERNELS
}  // namespace tensorflow
