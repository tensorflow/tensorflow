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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_device_array.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchDynamicStitchOp;

template <typename T>
struct LaunchDynamicStitchOp<CPUDevice, T> {
  static void launch(OpKernelContext* c, const int slice_size,
                     const int first_dim_size, OpInputList indices_inputs,
                     OpInputList data_inputs, Tensor* merged) {
    auto merged_flat = merged->flat_outer_dims<T>();
    for (int input_num = 0; input_num < indices_inputs.size(); input_num++) {
      const Tensor& indices = indices_inputs[input_num];
      auto indices_vec = indices.flat<int32>();
      const Tensor& data = data_inputs[input_num];
      auto data_flat =
          data.shaped<T, 2>({indices_vec.dimension(0), slice_size});

      if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
        T* merged_base = &merged_flat(0, 0);
        const T* data_base = &data_flat(0, 0);
        const size_t slice_bytes = slice_size * sizeof(T);
        for (int i = 0; i < indices_vec.size(); i++) {
          int32 index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(
              c, FastBoundsCheck(index, first_dim_size),
              errors::InvalidArgument("indices[", i, "] is out of range"));
          memcpy(merged_base + index * slice_size, data_base + i * slice_size,
                 slice_bytes);
        }
      } else {
        Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, slice_size);
        for (int i = 0; i < indices_vec.size(); i++) {
          // Copy slice data[i] to merged[indices[i]]
          Eigen::DSizes<Eigen::DenseIndex, 2> data_indices(i, 0);
          int32 index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(
              c, FastBoundsCheck(index, first_dim_size),
              errors::InvalidArgument("indices[", i, "] is out of range"));
          Eigen::DSizes<Eigen::DenseIndex, 2> merged_indices(index, 0);
          merged_flat.slice(merged_indices, sizes) =
              data_flat.slice(data_indices, sizes);
        }
      }
    }
  }
};

#if GOOGLE_CUDA

template <typename T>
struct DynamicStitchGPULaunch {
  static void Run(const GPUDevice& d, const int slice_size,
                  const int first_dim_size,
                  const CudaDeviceArrayStruct<int32>& indices_flat,
                  const CudaDeviceArrayStruct<const T*>& data_slice_ptrs,
                  T* output);
};

template <typename T>
struct LaunchDynamicStitchOp<GPUDevice, T> {
  static void launch(OpKernelContext* c, const int slice_size,
                     const int first_dim_size, OpInputList indices_inputs,
                     OpInputList data_inputs, Tensor* merged) {
    // createt two arrays that will be sent to CUDA device
    // one used as pointers to output's row id
    CudaDeviceArrayOnHost<int32> indices_flat(c, first_dim_size);
    // another used as pointers to the pointers of the head of each data_slice
    CudaDeviceArrayOnHost<const T*> data_slice_ptrs(c, first_dim_size);
    OP_REQUIRES_OK(c, indices_flat.Init());
    OP_REQUIRES_OK(c, data_slice_ptrs.Init());

    int ptr_num = 0;
    for (int input_num = 0; input_num < indices_inputs.size(); input_num++) {
      auto indices_vec = indices_inputs[input_num].flat<int32>();
      auto base_ptr = data_inputs[input_num].template flat<T>().data();
      for (int ind_num = 0; ind_num < indices_vec.size(); ind_num++) {
        indices_flat.Set(ptr_num, indices_vec(ind_num));
        // since each input data_slice is guranteed to be the same, we just
        // offset
        // base_ptr of each input tensor from data_inputs by a fixed amount.
        data_slice_ptrs.Set(
            ptr_num, const_cast<T*>(reinterpret_cast<const T*>(base_ptr) +
                                    slice_size * ind_num));
        ptr_num++;
      }
    }
    OP_REQUIRES_OK(c, indices_flat.Finalize());
    OP_REQUIRES_OK(c, data_slice_ptrs.Finalize());

    auto output = merged->template flat<T>().data();

    const GPUDevice& d = c->eigen_device<GPUDevice>();
    DynamicStitchGPULaunch<T>().Run(d, slice_size, first_dim_size,
                                    indices_flat.data(), data_slice_ptrs.data(),
                                    output);
    auto stream = c->op_device_context()->stream();
    OP_REQUIRES(c, stream->ok(),
                errors::Internal(
                    "Launch of gpu kernel for DynamicStitchGPULaunch failed"));
  }
};

#endif  // GOOGLE_CUDA

template <typename Device, typename T>
class DynamicStitchOp : public OpKernel {
 public:
  explicit DynamicStitchOp(OpKernelConstruction* c) : OpKernel(c) {
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
    OP_REQUIRES(
        c, c->num_inputs() > 0,
        errors::InvalidArgument("DynamicStitchOp: Must have some inputs"));
    OP_REQUIRES(c, c->num_inputs() % 2 == 0,
                errors::InvalidArgument(
                    "DynamicStitchOp: Must have even number of arguments"));
  }

  void Compute(OpKernelContext* c) override {
    // Find maximum index in the indices vectors
    OpInputList indices_inputs;
    OP_REQUIRES_OK(c, c->input_list("indices", &indices_inputs));

    int32 max_index = -1;
    for (const Tensor& indices : indices_inputs) {
      if (indices.NumElements() > 0) {
        Eigen::Tensor<int32, 0, Eigen::RowMajor> m =
            indices.flat<int32>().maximum();
        max_index = std::max(m(), max_index);
      }
    }

    const int first_dim_size = max_index + 1;

    // Validate that data[i].shape = indices[i].shape + constant
    OpInputList data_inputs;
    OP_REQUIRES_OK(c, c->input_list("data", &data_inputs));
    const Tensor& data0 = data_inputs[0];
    const Tensor& indices0 = indices_inputs[0];
    for (int input_num = 0; input_num < indices_inputs.size(); input_num++) {
      const Tensor& indices = indices_inputs[input_num];
      const Tensor& data = data_inputs[input_num];
      OP_REQUIRES(
          c, TensorShapeUtils::StartsWith(data.shape(), indices.shape()),
          errors::InvalidArgument("data[", input_num, "].shape = ",
                                  data.shape().DebugString(),
                                  " does not start with indices[", input_num,
                                  "].shape = ", indices.shape().DebugString()));
      OP_REQUIRES(
          c, input_num == 0 || SameExtraShape(data0, indices0, data, indices),
          errors::InvalidArgument(
              "Need data[0].shape[", indices0.dims(), ":] = data[", input_num,
              "].shape[", indices.dims(), ":], got data[0].shape = ",
              data0.shape().DebugString(), ", data[", input_num, "].shape = ",
              data.shape().DebugString(), ", indices[0].shape = ",
              indices0.shape().DebugString(), ", indices[", input_num,
              "].shape = ", indices.shape().DebugString()));
    }

    // Allocate result tensor of shape
    //   [first_dim_size] + data.shape[indices.dims:]
    TensorShape result_shape;
    result_shape.AddDim(first_dim_size);
    for (int d = indices0.dims(); d < data0.dims(); d++) {
      result_shape.AddDim(data0.dim_size(d));
    }
    Tensor* merged = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, &merged));
    // TODO(jeff): Currently we leave uninitialized any portions of
    // merged that aren't covered by an index in indices.  What should we do?
    if (first_dim_size > 0) {
      const int slice_size = merged->flat_outer_dims<T>().dimension(1);

      LaunchDynamicStitchOp<Device, T>::launch(
          c, slice_size, first_dim_size, indices_inputs, data_inputs, merged);
    }
  }

 private:
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
};

#define REGISTER_DYNAMIC_STITCH(type)                    \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOp<CPUDevice, type>)

TF_CALL_POD_STRING_TYPES(REGISTER_DYNAMIC_STITCH);
#undef REGISTER_DYNAMIC_STITCH

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_DYNAMIC_STITCH_SYCL(type)               \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_SYCL)       \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices")     \
                              .HostMemory("data")        \
                              .HostMemory("merged"),     \
                          DynamicStitchOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_DYNAMIC_STITCH_SYCL);
#undef REGISTER_DYNAMIC_STITCH_SYCL
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
#define REGISTER_DYNAMIC_STITCH_GPU(type)                \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_DYNAMIC_STITCH_GPU);
#undef REGISTER_DYNAMIC_STITCH_GPU

// dtypes that are not supported for current CUDA kernel use
// CPU kernel instead.
#define CALL_NOT_SUPPORTED_TYPES(m)                                          \
  TF_CALL_INTEGRAL_TYPES(m)                                                  \
  TF_CALL_half(m) TF_CALL_complex64(m) TF_CALL_complex128(m) TF_CALL_bool(m) \
      TF_CALL_string(m)

#define REGISTER_DYNAMIC_STITCH_GPU_NOT_SUPPORTED(type)  \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices")     \
                              .HostMemory("data")        \
                              .HostMemory("merged"),     \
                          DynamicStitchOp<CPUDevice, type>)

CALL_NOT_SUPPORTED_TYPES(REGISTER_DYNAMIC_STITCH_GPU_NOT_SUPPORTED);
#undef REGISTER_DYNAMIC_STITCH_GPU_NOT_SUPPORTED
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
