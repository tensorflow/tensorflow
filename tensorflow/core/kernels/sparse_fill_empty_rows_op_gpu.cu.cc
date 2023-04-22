/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/gpu_prim_helpers.h"
#include "tensorflow/core/kernels/sparse_fill_empty_rows_op.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#if GOOGLE_CUDA
#include "tensorflow/core/util/cuda_solvers.h"  // For ScratchSpace
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/rocm_solvers.h"
#include "tensorflow/stream_executor/rocm/rocm_activation.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace kernel_forward {
bool to_pointers(bool x) { return x; }
int32 to_pointers(int32 x) { return x; }
int64 to_pointers(int64 x) { return x; }
template <class T>
T* to_pointers(T* x) {
  return x;
}
template <class T>
typename T::PointerType to_pointers(T& x) {
  return x.data();
}
template <class T>
typename T::ConstPointerType to_pointers(const T& x) {
  return x.data();
}

template <typename Tindex, typename... CallerArgs, typename... KernelArgs>
Status wrap_kernel_call(void (*func)(KernelArgs...), const GPUDevice& device,
                        Tindex size, CallerArgs... args) {
  auto config = GetGpuLaunchConfig(size, device);
  return GpuLaunchKernel(func, config.block_count, config.thread_per_block, 0,
                         device.stream(), config, to_pointers(args)...);
}
};  // namespace kernel_forward

using kernel_forward::wrap_kernel_call;

namespace functor {

namespace {
template <typename To>
struct CastFunctor {
  template <typename From>
  __host__ __device__ To operator()(const From& value) const {
    return static_cast<To>(value);
  }
};

// Computes elements_per_row[0..dense_rows] and sets *rows_are_not_ordered to
// true if the indices are not ordered by row.
template <typename Tindex>
__global__ __launch_bounds__(1024) void CountElementsPerRowKernel(
    GpuLaunchConfig cfg, Tindex dense_rows, int rank, const Tindex* indices,
    Tindex* elements_per_row, int* rows_are_not_ordered,
    int* first_invalid_index) {
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    Tindex row = indices[i * rank];
    if (row < 0 || row >= dense_rows) {
      GpuAtomicMin(first_invalid_index, i);
      continue;
    }
    GpuAtomicAdd(&elements_per_row[row], 1);
    // Note that this only needs to compare rows, not columns, to satisfy the
    // row-major order invariant.
    if (i > 0 && row < indices[(i - 1) * rank]) {
      // TODO(benbarsdell): Replace this with atomic_ref::store when available.
      GpuAtomicMax(rows_are_not_ordered, 1);
    }
  }
}

template <typename Tindex>
__global__ __launch_bounds__(1024) void CopyRowIndicesKernel(
    GpuLaunchConfig cfg, int rank, const Tindex* indices, Tindex* row_indices) {
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    row_indices[i] = indices[i * rank];
  }
}

// Sets empty_row_indicator[row] to whether the row is empty.
template <typename Tindex>
__global__ __launch_bounds__(1024) void ComputeEmptyRowIndicatorKernel(
    GpuLaunchConfig cfg, const Tindex* elements_per_row,
    bool* empty_row_indicator) {
  GPU_1D_KERNEL_LOOP(row, cfg.virtual_thread_count) {
    empty_row_indicator[row] = elements_per_row[row] == 0;
  }
}

// Copies indices and values to output_indices and output_values, leaving space
// in the output for the new elements that will be inserted wherever there is an
// empty row.
template <typename T, typename Tindex>
__global__ __launch_bounds__(1024) void ScatterInputElementsKernel(
    GpuLaunchConfig cfg, Tindex dense_rows, int rank,
    const Tindex* input_index_map, const Tindex* indices, const T* values,
    const Tindex* num_new_rows_before, Tindex* output_indices, T* output_values,
    Tindex* reverse_index_map) {
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    Tindex input_i = input_index_map ? input_index_map[i] : i;
    Tindex row = indices[input_i * rank];
    Tindex output_i = i + num_new_rows_before[row];
    for (int dim = 0; dim < rank; ++dim) {
      output_indices[output_i * rank + dim] = indices[input_i * rank + dim];
    }
    output_values[output_i] = values[input_i];
    if (reverse_index_map) {
      reverse_index_map[input_i] = output_i;
    }
  }
}

// Sets the new elements (which correspond to the empty rows in the
// input) in output_indices and output_values.
template <typename T, typename Tindex>
__global__ __launch_bounds__(1024) void ScatterNewElementsKernel(
    GpuLaunchConfig cfg, int rank, const T* default_value,
    const Tindex* num_new_rows_through, const Tindex* input_row_ends,
    const bool* empty_row_indicator, Tindex* output_indices, T* output_values) {
  GPU_1D_KERNEL_LOOP(row, cfg.virtual_thread_count) {
    if (!empty_row_indicator[row]) continue;  // Only process empty rows
    Tindex input_i = (row == 0 ? 0 : input_row_ends[row - 1]);
    Tindex output_i = input_i + (row == 0 ? 0 : num_new_rows_through[row - 1]);
    for (int dim = 0; dim < rank; ++dim) {
      output_indices[output_i * rank + dim] = (dim == 0) ? row : 0;
    }
    output_values[output_i] = *default_value;
  }
}

}  // namespace

template <typename T, typename Tindex>
struct SparseFillEmptyRows<GPUDevice, T, Tindex> {
  Status operator()(OpKernelContext* context, const Tensor& default_value_t,
                    const Tensor& indices_t, const Tensor& values_t,
                    const Tensor& dense_shape_t,
                    typename AsyncOpKernel::DoneCallback done) {
    const int kEmptyRowIndicatorOutput = 2;

    const auto default_value = default_value_t.scalar<T>();
    const auto indices = indices_t.matrix<Tindex>();
    const auto values = values_t.vec<T>();
    const auto dense_shape = dense_shape_t.vec<Tindex>();

    const Tindex N = indices_t.shape().dim_size(0);
    const int rank = indices_t.shape().dim_size(1);
    const Tindex dense_rows = dense_shape(0);  // Must be on the host
    DataType index_type = DataTypeToEnum<Tindex>::value;
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    se::Stream* stream = context->op_device_context()->stream();
    if (!stream) return errors::Internal("No GPU stream available.");

    if (dense_rows == 0) {
      Tindex* output_indices;
      T* output_values;
      Tindex* reverse_index_map;
      TF_RETURN_IF_ERROR(AllocateOutputsExceptEmptyRowIndicator(
          context, N, rank, /*num_empty_rows=*/0, &output_indices,
          &output_values, &reverse_index_map));
      if (context->output_required(kEmptyRowIndicatorOutput)) {
        Tensor* unused = nullptr;
        TF_RETURN_IF_ERROR(context->allocate_output(kEmptyRowIndicatorOutput,
                                                    TensorShape({0}), &unused));
      }
      done();
      return Status::OK();
    }

    // The algorithm as currently implemented is summarized as follows:
    // 1) Compute elements_per_row (using GpuAtomicAdd).
    // 2) Compute input_row_ends (the end index of each row) by computing the
    //    prefix sum of elements_per_row.
    // 3) Compute empty_row_indicator = (elements_per_row == 0).
    // 4) Compute num_empty_rows_through (the no. empty rows up to and including
    //    each row) by computing the prefix sum of empty_row_indicator.
    // 5) Synchronize and allocate outputs (the sync is done implicitly by
    //    enqueueing the remainder of the computation onto the stream as a host
    //    callback).
    // 6) If rows are not ordered:
    //      Compute input_index_map by argsorting row indices.
    // 7) Scatter input elements into output_indices and output_values using
    //    input_index_map and num_empty_rows_through, leaving spaces for the
    //    new values that will be inserted.
    // 8) Scatter new default values into output_indices and output_values using
    //    num_new_rows_through, input_row_ends, and empty_row_indicator.

    // Summary of temporary allocations:
    //   Tindex elements_per_row[dense_rows]
    //   int rows_are_not_ordered[1]
    //   Tindex row_indices[N]      (if rows_are_not_ordered)
    //   Tindex input_index_map[N]  (if rows_are_not_ordered)
    //   Tindex input_row_ends[dense_rows]
    //   bool empty_row_indicator[dense_rows]
    //   Tindex num_empty_rows_through[dense_rows]
    //   Workspace for inclusive sums.
    //   Workspace for radix sort.

    Tensor elements_per_row_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        index_type, TensorShape({dense_rows}), &elements_per_row_t));
    auto elements_per_row = elements_per_row_t.flat<Tindex>();
    se::DeviceMemoryBase elements_per_row_gpu_memory(
        elements_per_row.data(), dense_rows * sizeof(Tindex));
    if (!stream
             ->ThenMemZero(&elements_per_row_gpu_memory,
                           dense_rows * sizeof(Tindex))
             .ok()) {
      return errors::Internal("Failed to zero elements_per_row");
    }
    Tensor rows_are_not_ordered_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT32, TensorShape({1}),
                                              &rows_are_not_ordered_t));
    auto rows_are_not_ordered_gpu = rows_are_not_ordered_t.flat<int>();
    se::DeviceMemoryBase rows_are_not_ordered_gpu_memory(
        rows_are_not_ordered_gpu.data(), sizeof(int));
    if (!stream->ThenMemZero(&rows_are_not_ordered_gpu_memory, sizeof(int))
             .ok()) {
      return errors::Internal("Failed to zero rows_are_not_ordered");
    }
    Tensor first_invalid_index_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT32, TensorShape({1}),
                                              &first_invalid_index_t));
    auto first_invalid_index_gpu = first_invalid_index_t.flat<int>();
    constexpr const int kAllIndicesValid = std::numeric_limits<int>::max();
    se::DeviceMemoryBase first_invalid_index_gpu_memory(
        first_invalid_index_gpu.data(), sizeof(int));
    if (!stream
             ->ThenMemset32(&first_invalid_index_gpu_memory, kAllIndicesValid,
                            sizeof(int))
             .ok()) {
      return errors::Internal("Failed to initialize first_invalid_index");
    }

    if (N > 0) {
      TF_RETURN_IF_ERROR(wrap_kernel_call(
          CountElementsPerRowKernel<Tindex>, /*device=*/device, /*size=*/N,
          dense_rows, rank, indices, elements_per_row, rows_are_not_ordered_gpu,
          first_invalid_index_gpu));
    }

    Tensor input_row_ends_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        index_type, TensorShape({dense_rows}), &input_row_ends_t));
    auto input_row_ends = input_row_ends_t.flat<Tindex>();

    TF_RETURN_IF_ERROR(GpuInclusivePrefixSum(context, /*size=*/dense_rows,
                                             /*input=*/elements_per_row.data(),
                                             /*output=*/input_row_ends.data()));

    Tensor empty_row_indicator_t;
    bool* empty_row_indicator;
    if (context->output_required(kEmptyRowIndicatorOutput)) {
      Tensor* empty_row_indicator_t_ptr = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(kEmptyRowIndicatorOutput,
                                                  TensorShape({dense_rows}),
                                                  &empty_row_indicator_t_ptr));
      empty_row_indicator = empty_row_indicator_t_ptr->vec<bool>().data();
    } else {
      TF_RETURN_IF_ERROR(context->allocate_temp(
          DT_BOOL, TensorShape({dense_rows}), &empty_row_indicator_t));
      empty_row_indicator = empty_row_indicator_t.vec<bool>().data();
    }

    TF_RETURN_IF_ERROR(wrap_kernel_call(ComputeEmptyRowIndicatorKernel<Tindex>,
                                        /*device=*/device, /*size=*/dense_rows,
                                        elements_per_row, empty_row_indicator));

    // For each row, the number of empty rows up to and including that row.
    Tensor num_empty_rows_through_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        index_type, TensorShape({dense_rows}), &num_empty_rows_through_t));
    auto num_empty_rows_through = num_empty_rows_through_t.flat<Tindex>();

    gpuprim::TransformInputIterator<Tindex, CastFunctor<Tindex>, bool*>
        empty_row_indicator_cast(empty_row_indicator, {});

    // The inclusive sum in CUB does not work do the right thing if
    // `empty_row_indicator` is passed in as a `bool *`.
    TF_RETURN_IF_ERROR(
        GpuInclusivePrefixSum(context, /*size=*/dense_rows,
                              /*input=*/empty_row_indicator_cast,
                              /*output=*/num_empty_rows_through.data()));

    ScratchSpace<Tindex> num_empty_rows_host(context, 1, /*on_host=*/true);
    if (!stream
             ->ThenMemcpy(num_empty_rows_host.mutable_data(),
                          se::DeviceMemoryBase(
                              num_empty_rows_through.data() + (dense_rows - 1),
                              sizeof(*num_empty_rows_host.data())),
                          sizeof(*num_empty_rows_host.data()))
             .ok()) {
      return errors::Internal("Failed to copy num_empty_rows to host");
    }

    ScratchSpace<int> rows_are_not_ordered_host(context, 1, /*on_host=*/true);
    if (!stream
             ->ThenMemcpy(rows_are_not_ordered_host.mutable_data(),
                          rows_are_not_ordered_gpu_memory,
                          sizeof(*rows_are_not_ordered_host.data()))
             .ok()) {
      return errors::Internal("Failed to copy rows_are_not_ordered to host");
    }

    ScratchSpace<int> first_invalid_index_host(context, 1, /*on_host=*/true);
    if (!stream
             ->ThenMemcpy(first_invalid_index_host.mutable_data(),
                          first_invalid_index_gpu_memory,
                          sizeof(*first_invalid_index_host.data()))
             .ok()) {
      return errors::Internal("Failed to copy first_invalid_index to host");
    }

    // We must wait for num_empty_rows and rows_are_not_ordered to be copied to
    // the host, so we enqueue the remainder of the computation onto the stream
    // asynchronously to avoid stalling execution.
    auto async_finish_computation =
        [this, context, kAllIndicesValid, index_type, N, rank, indices, values,
         default_value, dense_rows, num_empty_rows_host,
         rows_are_not_ordered_host, first_invalid_index_host,
         num_empty_rows_through_t, num_empty_rows_through, input_row_ends_t,
         input_row_ends, empty_row_indicator_t, empty_row_indicator,
         done]() -> void {
      DCHECK(done);  // Crash OK

      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = context->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      int first_invalid_index = *first_invalid_index_host.data();
      OP_REQUIRES_ASYNC(context, first_invalid_index == kAllIndicesValid,
                        errors::InvalidArgument("indices(", first_invalid_index,
                                                ", 0) is invalid."),
                        done);

      Tindex num_empty_rows = *num_empty_rows_host.data();

      Tindex* output_indices;
      T* output_values;
      Tindex* reverse_index_map;
      OP_REQUIRES_OK_ASYNC(
          context,
          AllocateOutputsExceptEmptyRowIndicator(
              context, N, rank, num_empty_rows, &output_indices, &output_values,
              &reverse_index_map),
          done);

      const GPUDevice& device = context->eigen_device<GPUDevice>();

      Tindex* input_index_map = nullptr;
      Tensor input_index_map_t;
      int rows_are_not_ordered = *rows_are_not_ordered_host.data();
      if (rows_are_not_ordered) {
        OP_REQUIRES_OK_ASYNC(context,
                             ArgSortByRows(context, device, N, rank, dense_rows,
                                           indices, &input_index_map_t),
                             done);
        input_index_map = input_index_map_t.vec<Tindex>().data();
      }

      if (N > 0) {
        OP_REQUIRES_OK_ASYNC(
            context,
            wrap_kernel_call(ScatterInputElementsKernel<T, Tindex>,
                             /*device=*/device, /*size=*/N, dense_rows, rank,
                             input_index_map, indices, values,
                             num_empty_rows_through, output_indices,
                             output_values, reverse_index_map),
            done);
      }

      OP_REQUIRES_OK_ASYNC(
          context,
          wrap_kernel_call(ScatterNewElementsKernel<T, Tindex>,
                           /*device=*/device, /*size=*/dense_rows, rank,
                           default_value, num_empty_rows_through,
                           input_row_ends, empty_row_indicator, output_indices,
                           output_values),
          done);

      done();
    };

    context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, async_finish_computation);
    return Status::OK();
  }

 private:
  Status AllocateOutputsExceptEmptyRowIndicator(
      OpKernelContext* context, Tindex N, int rank, Tindex num_empty_rows,
      Tindex** output_indices, T** output_values, Tindex** reverse_index_map) {
    Tensor* output_indices_t;
    const Tindex N_full = N + num_empty_rows;
    TensorShape output_indices_shape({N_full, rank});
    TF_RETURN_IF_ERROR(context->allocate_output(
        "output_indices", output_indices_shape, &output_indices_t));
    *output_indices = output_indices_t->matrix<Tindex>().data();

    Tensor* output_values_t;
    TF_RETURN_IF_ERROR(context->allocate_output(
        "output_values", TensorShape({N_full}), &output_values_t));
    *output_values = output_values_t->vec<T>().data();

    const int kReverseIndexMapOutput = 3;
    if (context->output_required(kReverseIndexMapOutput)) {
      Tensor* reverse_index_map_t = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(
          kReverseIndexMapOutput, TensorShape({N}), &reverse_index_map_t));
      *reverse_index_map = reverse_index_map_t->vec<Tindex>().data();
    } else {
      *reverse_index_map = nullptr;
    }
    return Status::OK();
  }

  Status ArgSortByRows(OpKernelContext* context, const GPUDevice& device,
                       Tindex N, int rank, Tindex dense_rows,
                       typename TTypes<Tindex>::ConstMatrix indices,
                       Tensor* input_index_map_t) {
    DataType index_type = DataTypeToEnum<Tindex>::value;
    // Extract row indices into separate array for use as keys for sorting.
    Tensor row_indices_t;
    TF_RETURN_IF_ERROR(
        context->allocate_temp(index_type, TensorShape({N}), &row_indices_t));
    auto row_indices = row_indices_t.flat<Tindex>();
    TF_RETURN_IF_ERROR(wrap_kernel_call(CopyRowIndicesKernel<Tindex>,
                                        /*device=*/device, /*size=*/N, rank,
                                        indices, row_indices));
    // Allocate input_index_map.
    TF_RETURN_IF_ERROR(context->allocate_temp(index_type, TensorShape({N}),
                                              input_index_map_t));
    Tindex* input_index_map = input_index_map_t->flat<Tindex>().data();
    return GpuRadixSort(context, /*size=*/N, /*keys_in=*/row_indices.data(),
                        /*keys_out=*/static_cast<Tindex*>(nullptr),
                        /*indices_in=*/static_cast<Tindex*>(nullptr),
                        /*indices_out=*/input_index_map,
                        /*num_bits=*/Log2Ceiling64(dense_rows));
  }
};

}  // namespace functor

#define DEFINE_INT64(T) \
  template struct functor::SparseFillEmptyRows<GPUDevice, T, int64>;
TF_CALL_POD_TYPES(DEFINE_INT64)
#undef DEFINE_INT64

namespace {

template <typename T, typename Tindex>
__global__ __launch_bounds__(1024) void GatherOriginalGradValuesKernel(
    GpuLaunchConfig cfg, const Tindex* reverse_index_map, const T* grad_values,
    T* d_values, bool* visited) {
  GPU_1D_KERNEL_LOOP(input_i, cfg.virtual_thread_count) {
    Tindex output_i = reverse_index_map[input_i];
    d_values[input_i] = grad_values[output_i];
    visited[output_i] = true;
  }
}

template <typename T, typename Tindex>
struct ZeroMaskedValues {
  ZeroMaskedValues(const bool* _mask, const T* _values)
      : mask(_mask), values(_values) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(Tindex i) const {
    return mask[i] ? T(0) : values[i];
  }
  const bool* mask;  // true means return zero instead of value
  const T* values;
};

}  // namespace

namespace functor {

template <typename T, typename Tindex>
struct SparseFillEmptyRowsGrad<GPUDevice, T, Tindex> {
  Status operator()(OpKernelContext* context,
                    typename TTypes<Tindex>::ConstVec reverse_index_map,
                    typename TTypes<T>::ConstVec grad_values,
                    typename TTypes<T>::Vec d_values,
                    typename TTypes<T>::Scalar d_default_value) {
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    const Tindex N = reverse_index_map.dimension(0);
    const Tindex N_full = grad_values.dimension(0);

    Tensor visited_t;
    TF_RETURN_IF_ERROR(
        context->allocate_temp(DT_BOOL, TensorShape({N_full}), &visited_t));
    auto visited = visited_t.vec<bool>();
    visited.device(device) = visited.constant(false);

    TF_RETURN_IF_ERROR(wrap_kernel_call(
        GatherOriginalGradValuesKernel<T, Tindex>, /*device=*/device,
        /*size=*/N, reverse_index_map, grad_values, d_values, visited));

    // Now we mask out the visited values and sum the remaining ones (which
    // correspond to the empty rows in the forward input) to compute
    // d_default_value.

    gpuprim::CountingInputIterator<Tindex, Tindex> counting_iterator(Tindex(0));
    ZeroMaskedValues<T, Tindex> mask_values_fn(visited.data(),
                                               grad_values.data());
    gpuprim::TransformInputIterator<T, decltype(mask_values_fn),
                                    decltype(counting_iterator), Tindex>
        transform_iterator(counting_iterator, mask_values_fn);

    std::size_t temp_storage_bytes = 0;
    auto gpuprim_status = gpuprim::DeviceReduce::Sum(
        /*temp_storage=*/nullptr, temp_storage_bytes,
        /*d_in=*/transform_iterator,
        /*d_out=*/d_default_value.data(),
        /*num_items=*/N_full,
        /*stream=*/device.stream());

    if (gpuprim_status != gpuSuccess) {
      return errors::Internal(
          "SparseFillEmptyRowsGrad: Could not launch "
          "gpuprim::DeviceReduce::Sum to calculate temp_storage_bytes, "
          "status: ",
          GpuGetErrorString(gpuprim_status));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
        &temp_storage));

    gpuprim_status = gpuprim::DeviceReduce::Sum(
        /*temp_storage=*/temp_storage.flat<int8>().data(), temp_storage_bytes,
        /*d_in=*/transform_iterator,
        /*d_out=*/d_default_value.data(),
        /*num_items=*/N_full,
        /*stream=*/device.stream());

    if (gpuprim_status != gpuSuccess) {
      return errors::Internal(
          "SparseFillEmptyRowsGrad: Could not launch "
          "gpuprim::DeviceReduce::Sum to sum values from originally-empty "
          "rows. temp_storage_bytes: ",
          temp_storage_bytes, ", status: ", GpuGetErrorString(gpuprim_status));
    }

    return Status::OK();
  }
};

}  // namespace functor

#define DEFINE_INT64(T) \
  template struct functor::SparseFillEmptyRowsGrad<GPUDevice, T, int64>;
TF_CALL_REAL_NUMBER_TYPES(DEFINE_INT64);
#undef DEFINE_INT64

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
