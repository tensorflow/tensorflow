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

#ifdef GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_device_array.h"
#endif // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#ifdef GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#endif // GOOGLE_CUDA


// Dynamic stitch function for CPUDevice.
template <typename T>
void DynamicStitchCPU(OpKernelContext* ctx,
                const int32 first_dim_size,
                const OpInputList& indices_inputs, const OpInputList& data_inputs,
                Tensor* merged) {
  auto merged_flat = merged->flat_outer_dims<T>();
  const int slice_size = merged_flat.dimension(1);
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
                ctx, FastBoundsCheck(index, first_dim_size),
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
                ctx, FastBoundsCheck(index, first_dim_size),
                errors::InvalidArgument("indices[", i, "] is out of range"));
        Eigen::DSizes<Eigen::DenseIndex, 2> merged_indices(index, 0);
        merged_flat.slice(merged_indices, sizes) =
                data_flat.slice(data_indices, sizes);
      }
    }
  }
}

#ifdef GOOGLE_CUDA

template <typename T>
void DynamicStitchGPUImpl(const Eigen::GpuDevice& gpu_device,
                          const int32 slice_size, const int32 first_dim_size,
                          const CudaDeviceArrayStruct<int>& input_indices,
                          const CudaDeviceArrayStruct<const T*>& input_ptrs,
                          T* output);


template <typename T>
void DynamicStitchGPU(OpKernelContext* c,
                        const int32 first_dim_size, const int32 data_elements_size,
                        const OpInputList& indices_inputs, const OpInputList& data_inputs,
                        Tensor* merged) {

  const int slice_size = merged->flat_outer_dims<T>().dimension(1);
  CudaDeviceArrayOnHost<int32> indices_flat(c, first_dim_size);
  CudaDeviceArrayOnHost<const T*> data_flat(c, data_elements_size);
  OP_REQUIRES_OK(c, indices_flat.Init());
  OP_REQUIRES_OK(c, data_flat.Init());
  // initialize the indices_flat (-1 represents missing indices)
  for (int i = 0; i < first_dim_size; ++i) {
    indices_flat.Set(i, -1);
  }

  int32 idx = 0;
  int32 base_size = 0;
  for (int i = 0; i < indices_inputs.size(); ++i) {
    auto indices_vec = indices_inputs[i].flat<int32>();
    auto data_ptr_base = data_inputs[i].template flat<T>().data();
    for(int j = 0; j < indices_vec.size(); ++j) {
      // indices_flat's values represent the indices located at data_flat.
      indices_flat.Set(indices_vec(j), base_size + j);
      data_flat.Set(idx, const_cast<T*>(reinterpret_cast<const T*>(data_ptr_base) +
                                        j * slice_size));
      ++idx;
    }
    base_size += indices_vec.size();
  }
  OP_REQUIRES_OK(c, indices_flat.Finalize());
  OP_REQUIRES_OK(c, data_flat.Finalize());

  auto output = merged->template flat<T>().data();
  DynamicStitchGPUImpl<T>(c->eigen_gpu_device(), slice_size, first_dim_size,
                          indices_flat.data(), data_flat.data(), output);
}
#endif // GOOGLE_CUDA


template <typename Device, class T>
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
    int32 data_elements_size = 0;
    for (const Tensor& indices : indices_inputs) {
      if (indices.NumElements() > 0) {
        Eigen::Tensor<int32, 0, Eigen::RowMajor> m =
            indices.flat<int32>().maximum();
        max_index = std::max(m(), max_index);
        data_elements_size += indices.NumElements();
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
#if GOOGLE_CUDA
      if (std::is_same<Device, GPUDevice>::value) {
        DynamicStitchGPU<T>(c, first_dim_size, data_elements_size, indices_inputs, data_inputs, merged);
        return;
      }
#endif  // GOOGLE_CUDA
      DynamicStitchCPU<T>(c, first_dim_size, indices_inputs, data_inputs, merged);
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

#if GOOGLE_CUDA
#define REGISTER_DYNAMIC_STITCH_GPU(type)                \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),     \
                          DynamicStitchOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_complex64(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_complex128(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_int64(REGISTER_DYNAMIC_STITCH_GPU);
#undef REGISTER_DYNAMIC_STITCH_GPU

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_DYNAMIC_STITCH_SYCL(type)               \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_SYCL)       \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices")     \
                              .HostMemory("data")        \
                              .HostMemory("merged"),     \
                          DynamicStitchOp<type>)

TF_CALL_POD_STRING_TYPES(REGISTER_DYNAMIC_STITCH_SYCL);
#undef REGISTER_DYNAMIC_STITCH_SYCL
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
