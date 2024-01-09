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

// See docs in ../ops/data_flow_ops.cc.

#include <algorithm>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/threadpool.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/gpu_device_array.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if !defined(PLUGGABLE_DEVICE_SUPPORTED_MACOS) && defined(__APPLE__) && \
    !defined(ANDROID) && !defined(__ANDROID__) &&                       \
    (!defined(TARGET_OS_IOS) || !TARGET_OS_IOS)
#define PLUGGABLE_DEVICE_SUPPORTED_MACOS 1
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <class T>
class DynamicStitchOpImplBase : public OpKernel {
 public:
  explicit DynamicStitchOpImplBase(OpKernelConstruction* c,
                                   const string& op_name)
      : OpKernel(c) {
    // Compute expected input signature
    const DataType dt = DataTypeToEnum<T>::v();
    const int n = c->num_inputs() / 2;
    DataTypeVector expected;
    for (int i = 0; i < n; i++) {
      expected.push_back(DT_INT32);
    }
    for (int i = 0; i < n; i++) {
      expected.push_back(dt);
    }
    OP_REQUIRES_OK(c, c->MatchSignature(expected, {dt}));
    OP_REQUIRES(c, c->num_inputs() > 0,
                errors::InvalidArgument(op_name + ": Must have some inputs"));
    OP_REQUIRES(c, c->num_inputs() % 2 == 0,
                errors::InvalidArgument(
                    op_name + ": Must have even number of arguments"));
  }

 protected:
  // Check if data0.shape[indices0.dims():] == data1.shape[indices1.dims():]
  static bool SameExtraShape(const Tensor& data0, const Tensor& indices0,
                             const Tensor& data1, const Tensor& indices1) {
    const int extra0 = data0.dims() - indices0.dims();
    const int extra1 = data1.dims() - indices1.dims();
    if (extra0 != extra1) return false;
    for (int i = 0; i < extra0; i++) {
      if (data0.dim_size(indices0.dims() + i) !=
          data1.dim_size(indices1.dims() + i)) {
        return false;
      }
    }
    return true;
  }

  void CheckArgsAndAllocateResult(OpKernelContext* c,
                                  OpInputList* indices_inputs,
                                  OpInputList* data_inputs, int* first_dim_size,
                                  int* data_elements_size,
                                  Tensor** result_ptr) {
    // Find maximum index in the indices vectors
    OP_REQUIRES_OK(c, c->input_list("indices", indices_inputs));

    int32_t max_index = -1;
    if (data_elements_size) {
      *data_elements_size = 0;
    }
    for (const Tensor& indices : *indices_inputs) {
      if (indices.NumElements() > 0) {
        Eigen::Tensor<int32, 0, Eigen::RowMajor> m =
            indices.flat<int32>().maximum();
        max_index = std::max(m(), max_index);
      }
      if (data_elements_size) {
        *data_elements_size += indices.NumElements();
      }
    }

    *first_dim_size = max_index + 1;

    for (const Tensor& indices : *indices_inputs) {
      auto indices_vec = indices.flat<int32>();

      for (int i = 0; i < indices_vec.size(); i++) {
        int32_t index = internal::SubtleMustCopy(indices_vec(i));
        OP_REQUIRES(
            c, FastBoundsCheck(index, *first_dim_size),
            errors::InvalidArgument("indices[", i, "] is out of range"));
      }
    }

    // Validate that data[i].shape = indices[i].shape + constant
    OP_REQUIRES_OK(c, c->input_list("data", data_inputs));
    const Tensor& data0 = (*data_inputs)[0];
    const Tensor& indices0 = (*indices_inputs)[0];
    for (int input_num = 0; input_num < indices_inputs->size(); input_num++) {
      const Tensor& indices = (*indices_inputs)[input_num];
      const Tensor& data = (*data_inputs)[input_num];
      OP_REQUIRES(
          c, TensorShapeUtils::StartsWith(data.shape(), indices.shape()),
          errors::InvalidArgument("data[", input_num,
                                  "].shape = ", data.shape().DebugString(),
                                  " does not start with indices[", input_num,
                                  "].shape = ", indices.shape().DebugString()));
      OP_REQUIRES(
          c, input_num == 0 || SameExtraShape(data0, indices0, data, indices),
          errors::InvalidArgument(
              "Need data[0].shape[", indices0.dims(), ":] = data[", input_num,
              "].shape[", indices.dims(),
              ":], got data[0].shape = ", data0.shape().DebugString(),
              ", data[", input_num, "].shape = ", data.shape().DebugString(),
              ", indices[0].shape = ", indices0.shape().DebugString(),
              ", indices[", input_num,
              "].shape = ", indices.shape().DebugString()));
    }

    // Allocate result tensor of shape
    //   [*first_dim_size] + data.shape[indices.dims:]
    TensorShape result_shape;
    OP_REQUIRES_OK(c, result_shape.AddDimWithStatus(*first_dim_size));
    for (int d = indices0.dims(); d < data0.dims(); d++) {
      OP_REQUIRES_OK(c, result_shape.AddDimWithStatus(data0.dim_size(d)));
    }
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, result_ptr));
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
void DynamicStitchGPUImpl(const Eigen::GpuDevice& gpu_device,
                          const int32_t slice_size,
                          const int32_t first_dim_size,
                          const GpuDeviceArrayStruct<int>& input_indices,
                          const GpuDeviceArrayStruct<const T*>& input_ptrs,
                          T* output);
#define REGISTER_GPU(T)                                           \
  extern template void DynamicStitchGPUImpl(                      \
      const Eigen::GpuDevice& gpu_device, const int32 slice_size, \
      const int32 first_dim_size,                                 \
      const GpuDeviceArrayStruct<int32>& input_indices,           \
      const GpuDeviceArrayStruct<const T*>& input_ptrs, T* output);

TF_CALL_int32(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

template <class T>
class DynamicStitchOpGPU : public DynamicStitchOpImplBase<T> {
 public:
  explicit DynamicStitchOpGPU(OpKernelConstruction* c)
      : DynamicStitchOpImplBase<T>(c, "DynamicStitchOp") {}

  void Compute(OpKernelContext* c) override {
    OpInputList indices_inputs;
    OpInputList data_inputs;
    int first_dim_size;
    int data_elements_size;
    Tensor* merged = nullptr;
    this->CheckArgsAndAllocateResult(c, &indices_inputs, &data_inputs,
                                     &first_dim_size, &data_elements_size,
                                     &merged);
    if (!c->status().ok()) {
      // Avoid segmentation faults if merged cannot be allocated and an error is
      // passed back in the context.
      return;
    }

    if (first_dim_size > 0) {
      // because the collision requirements, we have to deal with
      // collision first before send data to gpu kernel.
      // TODO(ekelsen): Instead of doing a serial scan on the CPU to pick the
      // last of duplicated indices, it could instead be done of the GPU
      // implicitly using atomics to make sure the last index is the final
      // write.
      const int slice_size = merged->flat_outer_dims<T>().dimension(1);
      GpuDeviceArrayOnHost<int32> indices_flat(c, first_dim_size);
      GpuDeviceArrayOnHost<const T*> data_flat(c, data_elements_size);
      OP_REQUIRES_OK(c, indices_flat.Init());
      OP_REQUIRES_OK(c, data_flat.Init());
      // initialize the indices_flat (-1 represents missing indices)
      for (int i = 0; i < first_dim_size; ++i) {
        indices_flat.Set(i, -1);
      }

      // data_flat index
      int32_t idx = 0;
      // sum of indices_inputs[i].NumElements() for compute indices_flat value.
      int32_t base_size = 0;
      for (int i = 0; i < indices_inputs.size(); ++i) {
        auto indices_vec = indices_inputs[i].flat<int32>();
        auto data_ptr_base = data_inputs[i].template flat<T>().data();
        for (int j = 0; j < indices_vec.size(); ++j) {
          // indices_flat's indices represent the indices of output.
          // indices_flat's values represent the indices of input_data where the
          // data located.
          indices_flat.Set(indices_vec(j), base_size + j);
          data_flat.Set(
              idx, const_cast<T*>(reinterpret_cast<const T*>(data_ptr_base) +
                                  j * slice_size));
          ++idx;
        }
        base_size += indices_vec.size();
      }
      OP_REQUIRES_OK(c, indices_flat.Finalize());
      OP_REQUIRES_OK(c, data_flat.Finalize());

      auto merged_flat = merged->template flat<T>();
      functor::SetZeroFunctor<GPUDevice, T> f;
      f(c->eigen_device<GPUDevice>(), merged_flat);
      auto output = merged_flat.data();
      DynamicStitchGPUImpl<T>(c->eigen_gpu_device(), slice_size, first_dim_size,
                              indices_flat.data(), data_flat.data(), output);
    }
  }
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <class T, bool Parallel>
class DynamicStitchOpImplCPU : public DynamicStitchOpImplBase<T> {
 public:
  explicit DynamicStitchOpImplCPU(OpKernelConstruction* c)
      : DynamicStitchOpImplBase<T>(
            c, (Parallel ? "ParallelDynamicStitchOp" : "DynamicStitchOp")) {}

  void Compute(OpKernelContext* c) override {
    OpInputList indices_inputs;
    OpInputList data_inputs;
    int first_dim_size;
    Tensor* merged = nullptr;
    this->CheckArgsAndAllocateResult(c, &indices_inputs, &data_inputs,
                                     &first_dim_size, nullptr, &merged);
    if (!c->status().ok()) {
      // Avoid segmentation faults if merged cannot be allocated and an error is
      // passed back in the context.
      return;
    }

    if (first_dim_size > 0) {
      functor::SetZeroFunctor<CPUDevice, T> f;
      f(c->eigen_device<CPUDevice>(), merged->template flat<T>());
      auto merged_flat = merged->flat_outer_dims<T>();
      // slice_size must not be stored as int for cases of tensors over 2GB.
      const auto slice_size = merged_flat.dimension(1);
      const size_t slice_bytes = slice_size * sizeof(T);
      auto OnInputNumber = [&](int input_num) {
        const Tensor& indices = indices_inputs[input_num];
        auto indices_vec = indices.flat<int32>();
        const Tensor& data = data_inputs[input_num];
        auto data_flat =
            data.shaped<T, 2>({indices_vec.dimension(0), slice_size});

        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          T* merged_base = merged_flat.data();
          const T* data_base = data_flat.data();
          for (int i = 0; i < indices_vec.size(); i++) {
            int32_t index = internal::SubtleMustCopy(indices_vec(i));
            memcpy(merged_base + index * slice_size, data_base + i * slice_size,
                   slice_bytes);
          }
        } else {
          Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, slice_size);
          for (int i = 0; i < indices_vec.size(); i++) {
            // Copy slice data[i] to merged[indices[i]]
            Eigen::DSizes<Eigen::DenseIndex, 2> data_indices(i, 0);
            int32_t index = internal::SubtleMustCopy(indices_vec(i));
            Eigen::DSizes<Eigen::DenseIndex, 2> merged_indices(index, 0);
            merged_flat.slice(merged_indices, sizes) =
                data_flat.slice(data_indices, sizes);
          }
        }
      };
      if (Parallel &&
          c->device()->tensorflow_cpu_worker_threads()->num_threads > 1) {
        auto thread_pool =
            c->device()->tensorflow_cpu_worker_threads()->workers;
        size_t total_indices_size = 0;
        for (int input_num = 0; input_num < indices_inputs.size();
             ++input_num) {
          total_indices_size += indices_inputs[input_num].NumElements();
        }
        const double avg_indices_size =
            static_cast<double>(total_indices_size) / indices_inputs.size();
        auto bytes_processed = slice_bytes * avg_indices_size;
        auto LoopBody = [&](int first, int last) {
          for (int input_num = first; input_num < last; ++input_num) {
            OnInputNumber(input_num);
          }
        };
        thread_pool->ParallelFor(indices_inputs.size(), bytes_processed,
                                 LoopBody);
      } else {
        for (int input_num = 0; input_num < indices_inputs.size();
             input_num++) {
          OnInputNumber(input_num);
        }
      }
    }
  }
};

// Using inheritance rather than a typedef so that these classes might have more
// functionality later.

template <typename T>
struct DynamicStitchOpCPU : DynamicStitchOpImplCPU<T, false> {
  using DynamicStitchOpImplCPU<T, false>::DynamicStitchOpImplCPU;
};

template <typename T>
struct ParallelDynamicStitchOpCPU : DynamicStitchOpImplCPU<T, true> {
  using DynamicStitchOpImplCPU<T, true>::DynamicStitchOpImplCPU;
};

#define REGISTER_DYNAMIC_STITCH(type)                    \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOpCPU<type>)      \
  REGISTER_KERNEL_BUILDER(Name("ParallelDynamicStitch")  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          ParallelDynamicStitchOpCPU<type>)

TF_CALL_POD_STRING_TYPES(REGISTER_DYNAMIC_STITCH);
TF_CALL_variant(REGISTER_DYNAMIC_STITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_DYNAMIC_STITCH);
#undef REGISTER_DYNAMIC_STITCH

#define REGISTER_PARALLEL_DYNAMIC_STITCH(type)           \
  REGISTER_KERNEL_BUILDER(Name("ParallelDynamicStitch")  \
                              .Device(DEVICE_DEFAULT)    \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices")     \
                              .HostMemory("data")        \
                              .HostMemory("merged"),     \
                          ParallelDynamicStitchOpCPU<type>)

TF_CALL_int32(REGISTER_PARALLEL_DYNAMIC_STITCH);
TF_CALL_int64(REGISTER_PARALLEL_DYNAMIC_STITCH);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_PARALLEL_DYNAMIC_STITCH);
TF_CALL_COMPLEX_TYPES(REGISTER_PARALLEL_DYNAMIC_STITCH);
#undef REGISTER_PARALLEL_DYNAMIC_STITCH

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_DYNAMIC_STITCH_GPU(type)                \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOpGPU<type>)

TF_CALL_int32(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_int64(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_DYNAMIC_STITCH_GPU);
#undef REGISTER_DYNAMIC_STITCH_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if defined(PLUGGABLE_DEVICE_SUPPORTED_MACOS)
#define REGISTER_DYNAMIC_STITCH_DEFAULT_DEVICE(type)     \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_DEFAULT)    \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices")     \
                              .HostMemory("data")        \
                              .HostMemory("merged"),     \
                          DynamicStitchOpCPU<type>)

TF_CALL_int32(REGISTER_DYNAMIC_STITCH_DEFAULT_DEVICE);
TF_CALL_int64(REGISTER_DYNAMIC_STITCH_DEFAULT_DEVICE);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_DYNAMIC_STITCH_DEFAULT_DEVICE);
TF_CALL_COMPLEX_TYPES(REGISTER_DYNAMIC_STITCH_DEFAULT_DEVICE);
#undef REGISTER_DYNAMIC_STITCH_DEFAULT_DEVICE
#endif

}  // namespace tensorflow
