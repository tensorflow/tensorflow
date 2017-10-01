/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/gather_functor.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

#ifdef GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_device_array.h"
#endif  // GOOGLE_CUDA

#include <stdlib.h>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Data structure tracking repeated slices
struct SliceBitSet {
  SliceBitSet(int64 size) {
    data = static_cast<char*>(port::Malloc(size));
    memset(data, 0, size);
  }
  ~SliceBitSet() { port::Free(data); }
  bool IsSet(int64 bit) { return (data[bit >> 3] >> (bit % 8)) & 1; }
  void SetBit(int64 bit) { data[bit >> 3] |= (1 << (bit % 8)); }
  char* data;
};

template <typename Device, typename T, typename Index>
class GatherDisjointOpBase : public OpKernel {
 public:
  explicit GatherDisjointOpBase(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("reverse_order", &reverse_order_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& params = c->input(0);

    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));

    OpInputList indices_inputs;
    OP_REQUIRES_OK(c, c->input_list("indices", &indices_inputs));

    // Check that we have enough index space
    const int64 gather_dim_size = params.dim_size(0);
    OP_REQUIRES(c, gather_dim_size <= std::numeric_limits<Index>::max(),
                errors::InvalidArgument(
                    "params.shape[0] too large for ",
                    DataTypeString(DataTypeToEnum<Index>::v()), " indexing: ",
                    gather_dim_size, " > ", std::numeric_limits<Index>::max()));

    int input_num = 0, step = 1, end = indices_inputs.size();
    if (reverse_order_) {
      input_num = indices_inputs.size() - 1;
      step = -1;
      end = -1;
    }

    // We use a bitset to know which slices were already
    // gathered
    const int64 bitset_size = (gather_dim_size + 7) >> 3;
    SliceBitSet used_slices(bitset_size);

    for (; input_num != end; input_num += step) {
      // Allocate output tensor
      TensorShape result_shape;
      int64 inner_size = 1;
      result_shape.AppendShape(indices_inputs[input_num].shape());
      for (int j = 1; j < params.dims(); ++j) {
        result_shape.AddDim(params.dim_size(j));
        inner_size *= params.dim_size(j);
      }

      Tensor* out = nullptr;
      OP_REQUIRES_OK(c, c->allocate_output(input_num, result_shape, &out));

      int64 N = indices_inputs[input_num].NumElements();
      if (N > 0 && inner_size > 0) {
        // Gather the values in the output tensor
        auto params_flat =
            params.shaped<T, 3>({1, gather_dim_size, inner_size});
        auto indices_flat = indices_inputs[input_num].flat<Index>();
        auto out_flat = out->shaped<T, 3>({1, N, inner_size});

        functor::GatherFunctor<Device, T, Index> functor;
        int64 bad_i = functor(c->eigen_device<Device>(), params_flat,
                              indices_flat, out_flat);

        OP_REQUIRES(
            c, bad_i < 0,
            errors::InvalidArgument(
                "indices",
                SliceDebugString(indices_inputs[input_num].shape(), bad_i),
                " = ", indices_flat(bad_i), " is not in [0, ", gather_dim_size,
                ")"));

        // Fill repeated slices with value 0
        auto output = out->shaped<T, 2>({N, inner_size});
        this->HandleRepeatedSlices(c, &used_slices, indices_flat,
                                   reverse_order_, output);
      }
    }
  }

 protected:
  // We provide different implementations for CPU and GPU.
  virtual void HandleRepeatedSlices(OpKernelContext* c,
                                    SliceBitSet* used_slices,
                                    typename TTypes<Index>::ConstFlat indices,
                                    bool reverse_order,
                                    typename TTypes<T, 2>::Tensor output) = 0;

 private:
  bool reverse_order_;
};

#if GOOGLE_CUDA

template <typename T>
void GatherDisjointOpGPUImpl(const Eigen::GpuDevice& gpu_device,
                             const int64 first_dim_size,
                             const int64 slice_elems,
                             const CudaDeviceArrayStruct<int8>& zero_indicator,
                             T* output);

template <typename T, typename Index>
class GatherDisjointOpGPU : public GatherDisjointOpBase<GPUDevice, T, Index> {
 public:
  explicit GatherDisjointOpGPU(OpKernelConstruction* c)
      : GatherDisjointOpBase<GPUDevice, T, Index>(c) {}

 protected:
  void HandleRepeatedSlices(OpKernelContext* c, SliceBitSet* used_slices,
                            typename TTypes<Index>::ConstFlat indices,
                            bool reverse_order,
                            typename TTypes<T, 2>::Tensor output) {
    const int64 out_size = output.size();
    if (out_size > 0) {
      const int64 indices_size = indices.size();
      const int64 slice_elems = static_cast<int64>(output.dimension(1));

      CudaDeviceArrayOnHost<int8> zero_indicator(c, indices_size);
      OP_REQUIRES_OK(c, zero_indicator.Init());

      // initialize the zero_indicator
      for (int i = 0; i < indices_size; ++i) {
        zero_indicator.Set(i, 0);
      }

      int64 i = 0, step = 1, end = indices_size;
      if (reverse_order) {
        i = indices_size - 1;
        step = -1;
        end = -1;
      }
      // Find all the repeated indices in a loop on the CPU.
      // This is similar to the GPU implementation of Dynamic Stitch,
      // which resolves collisions in a CPU loop before calling the kernel.
      for (; i != end; i += step) {
        const Index index = internal::SubtleMustCopy(indices(i));
        // Check if the index was already used and if yes,
        // mark the location.
        if (used_slices->IsSet(static_cast<int64>(index)))
          zero_indicator.Set(i, 1);
        else
          used_slices->SetBit(static_cast<int64>(index));
      }
      OP_REQUIRES_OK(c, zero_indicator.Finalize());

      GatherDisjointOpGPUImpl<T>(c->eigen_gpu_device(), indices_size,
                                 slice_elems, zero_indicator.data(),
                                 output.data());
    }
  }
};

#endif  // GOOGLE_CUDA

template <typename T, typename Index>
class GatherDisjointOpCPU : public GatherDisjointOpBase<CPUDevice, T, Index> {
 public:
  explicit GatherDisjointOpCPU(OpKernelConstruction* c)
      : GatherDisjointOpBase<CPUDevice, T, Index>(c) {}

 protected:
  void HandleRepeatedSlices(OpKernelContext* c, SliceBitSet* used_slices,
                            typename TTypes<Index>::ConstFlat indices,
                            bool reverse_order,
                            typename TTypes<T, 2>::Tensor output) {
    T* out_base = &output(0, 0);
    // Compute the inner size of the output
    const int64 slice_elems = static_cast<int64>(output.dimension(1));
    // Compute slice_bytes
    const size_t slice_bytes = slice_elems * sizeof(T);
    int64 i = 0, step = 1, end = static_cast<int64>(indices.size());
    if (reverse_order) {
      i = end - 1;
      step = -1;
      end = -1;
    }
    for (; i != end; i += step) {
      // Grab the index. We don't check the validity as it was already
      // checked before.
      const Index index = internal::SubtleMustCopy(indices(i));
      // Check if the index was already used and if yes,
      // overwrite the slice with 0.
      if (used_slices->IsSet(static_cast<int64>(index)))
        memset(out_base + i * slice_elems, 0, slice_bytes);
      else
        used_slices->SetBit(static_cast<int64>(index));
    }
  }
};

#define REGISTER_GATHER_DISJOINT_CPU(type, index_type)                 \
  REGISTER_KERNEL_BUILDER(Name("GatherDisjoint")                       \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("Tparams")         \
                              .TypeConstraint<index_type>("Tindices"), \
                          GatherDisjointOpCPU<type, index_type>);

#define REGISTER_GATHER_DISJOINT_CPU_ALL_INDICES(type) \
  REGISTER_GATHER_DISJOINT_CPU(type, int32);           \
  REGISTER_GATHER_DISJOINT_CPU(type, int64)

// Registration of the CPU implementations.
TF_CALL_NUMBER_TYPES(REGISTER_GATHER_DISJOINT_CPU_ALL_INDICES);

#undef REGISTER_GATHER_DISJOINT_CPU_ALL_INDICES
#undef REGISTER_GATHER_DISJOINT_CPU

#if GOOGLE_CUDA

// Registration of the GPU implementations.
#define REGISTER_GATHER_DISJOINT_GPU(type, index_type)                \
  REGISTER_KERNEL_BUILDER(Name("GatherDisjoint")                      \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("Tparams")        \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("indices"),                 \
                          GatherDisjointOpGPU<type, index_type>);

#define REGISTER_GATHER_DISJOINT_GPU_ALL_INDICES(type) \
  REGISTER_GATHER_DISJOINT_GPU(type, int32);           \
  REGISTER_GATHER_DISJOINT_GPU(type, int64)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GATHER_DISJOINT_GPU_ALL_INDICES);
TF_CALL_complex64(REGISTER_GATHER_DISJOINT_GPU_ALL_INDICES);
TF_CALL_complex128(REGISTER_GATHER_DISJOINT_GPU_ALL_INDICES);

#undef REGISTER_GATHER_DISJOINT_GPU_ALL_INDICES
#undef REGISTER_GATHER_DISJOINT_GPU

#endif  // GOOGLE_CUDA
}  // namespace tensorflow
