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

namespace tensorflow {

template <class T>
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
                          DynamicStitchOp<type>)

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
                          DynamicStitchOp<type>)

TF_CALL_ALL_TYPES(REGISTER_DYNAMIC_STITCH_SYCL);
#undef REGISTER_DYNAMIC_STITCH_SYCL
#endif // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
#define REGISTER_DYNAMIC_STITCH_GPU(type)                \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices")     \
                              .HostMemory("data")        \
                              .HostMemory("merged"),     \
                          DynamicStitchOp<type>)

TF_CALL_POD_STRING_TYPES(REGISTER_DYNAMIC_STITCH_GPU);
#undef REGISTER_DYNAMIC_STITCH_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
