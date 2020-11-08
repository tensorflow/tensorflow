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

// See core/ops/sparse_ops.cc for documentation.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

__global__ void SparseIndexValidKernel(const int64* sparse_indices, size_t num_elements,
                                       const int64* shape_vec, size_t num_dims,
                                       const int64* order_vec, size_t num_order,
                                       int* error_flag) {
  bool valid = true;
  bool different = false;
  bool increasing = true;
  auto idx_fn = [=](int num_idx, int dim_idx) -> int64 {
    return sparse_indices[num_idx * num_dims + dim_idx];
  };
  GPU_1D_KERNEL_LOOP(n, num_elements) {
    if (n == 0) {
      for (int di = 0; di < num_dims; ++di) {
        if (idx_fn(n, di) < 0 || idx_fn(n, di) >= shape_vec[di]) valid = false;
      }
      different = true;
    } else {
      for (int di = 0; di < num_dims; ++di) {
        if (idx_fn(n, di) < 0 || idx_fn(n, di) >= shape_vec[di]) valid = false;
        int64 diff = idx_fn(n, order_vec[di]) - idx_fn(n - 1, order_vec[di]);
        if (diff > 0) different = true;
        if (!different && diff < 0) increasing = false;
      }
    }
    if (!valid)
      error_flag[0] = !valid;
    if (!increasing)
      error_flag[1] = !increasing;
    if (!different)
      error_flag[2] = !different;
    if (!valid || !increasing || !different) {
      error_flag[3] = n;
      break;
    }
  }
}

template<typename T>
__global__ void SparseToDenseKernel(const int64* sparse_indices, const T* sparse_values,
                                    T* dense_out, size_t num_elements,
                                    const int64* strides_vec, const int64* dim_vec,
                                    size_t num_dims, int* error_flag) {
  GPU_1D_KERNEL_LOOP(i, num_elements) {
    bool invalid_dims = false;
    int64 ix = 0;
    for (int d = 0; d < num_dims; ++d) {
      const int64 ix_n_d = internal::SubtleMustCopy(sparse_indices[i * num_dims + d]);
      if (!FastBoundsCheck(ix_n_d, dim_vec[d])) {
        invalid_dims = true;
      }
      ix += strides_vec[d] * ix_n_d;
    }
    if (invalid_dims) {
      error_flag[0] = 1;
      break;
    }
    dense_out[ix] = sparse_values[i];
  }
}

namespace functor {

template<typename T, typename Index>
struct SparseToDenseFunctor<GPUDevice, T, Index> {
  void operator()(OpKernelContext* c, const Tensor& indices,
                  const Tensor& output_shape, const Tensor& sparse_values,
                  const Tensor& default_value, bool validate_indices) {
    const GPUDevice& d = c->eigen_device<GPUDevice>();
    const int64 num_elems = indices.dims() > 0 ? indices.dim_size(0) : 1;
    const int64 num_dims = indices.dims() > 1 ? indices.dim_size(1) : 1;

    auto out_shape_d_vec = output_shape.flat<Index>();
    Tensor out_shape_host(output_shape.dtype(), output_shape.shape());
    auto out_shape_host_vec = out_shape_host.flat<Index>();

    // we need output_shape immediately to allocate output tensor
    cudaMemcpy(out_shape_host_vec.data(), out_shape_d_vec.data(),
               output_shape.TotalBytes(), cudaMemcpyDeviceToHost);
    TensorShape output_tensor_shape;
    OP_REQUIRES_OK(c, TensorShapeUtils::MakeShape(out_shape_host_vec.data(),
                                                  out_shape_host_vec.size(),
                                                  &output_tensor_shape));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_tensor_shape, &output));

    TensorShape ix_shape({num_elems, num_dims});
    Tensor indices_shaped;
    if (indices.dtype() == DT_INT64) {
      indices_shaped = Tensor(DT_INT64, ix_shape);
      CHECK(indices_shaped.CopyFrom(indices, ix_shape));
    } else {
      OP_REQUIRES_OK(c, c->allocate_temp(DT_INT64, ix_shape, &indices_shaped));
      indices_shaped.matrix<int64>().device(d) =
          indices.shaped<Index, 2>(ix_shape.dim_sizes()).template cast<int64>();
    }

    // If we received a scalar, we'll need to create a new
    // tensor with copies of the values as a vec.
    // TODO(ebrevdo): find a way to avoid this temp allocation.
    Tensor sparse_values_b;
    if (TensorShapeUtils::IsScalar(sparse_values.shape())) {
      OP_REQUIRES_OK(
          c, c->allocate_temp(DataTypeToEnum<T>::value,
                              TensorShape({num_elems}), &sparse_values_b));
      auto sparse_val_b_vec = sparse_values_b.vec<T>();
      Eigen::Sizes<1> single;
      sparse_val_b_vec.device(d) =
          sparse_values.scalar<T>().reshape(single).broadcast(sparse_val_b_vec.dimensions());
    } else {
      sparse_values_b = sparse_values;
    }

    auto dim_vec_bytes = sizeof(int64) * num_dims;
    std::vector<int64> dim_vec(num_dims);
    auto* dim_vec_data = output_tensor_shape.dim_sizes().data();
    auto* d_dim_ptr = reinterpret_cast<int64*>(d.allocate(dim_vec_bytes));
    d.memcpyHostToDevice(d_dim_ptr, dim_vec_data, dim_vec_bytes);

    if (validate_indices) {
      gtl::InlinedVector<int64, 8> order(output->shape().dims());
      std::iota(order.begin(), order.end(), 0);
      auto* d_order_ptr = reinterpret_cast<int64*>(d.allocate(dim_vec_bytes));
      d.memcpyHostToDevice(d_order_ptr, order.data(), dim_vec_bytes);

      auto flag_bytes = sizeof(size_t) * 4;
      auto* d_error_flag = reinterpret_cast<int*>(d.allocate(flag_bytes));
      d.memset(d_error_flag, 0, flag_bytes);

      GpuLaunchConfig cfg = GetGpuLaunchConfig(num_elems, d);
      TF_CHECK_OK(GpuLaunchKernel(SparseIndexValidKernel, cfg.block_count,
                                  cfg.thread_per_block, 0, d.stream(),
                                  indices_shaped.matrix<int64>().data(),
                                  num_elems, d_dim_ptr, num_dims,
                                  d_order_ptr, num_dims, d_error_flag));

      gpuEvent_t copy_done;
      gpuEventCreate(&copy_done);
      auto* error_flag = (int*) cpu_allocator()->AllocateRaw(sizeof(size_t), flag_bytes);
      d.memcpyDeviceToHost(error_flag, d_error_flag, flag_bytes);
      TF_OP_REQUIRES_CUDA_SUCCESS(c, gpuEventRecord(copy_done, d.stream()));
      TF_OP_REQUIRES_CUDA_SUCCESS(c, gpuEventSynchronize(copy_done));

      bool valid = !error_flag[0];
      bool increasing = !error_flag[1];
      bool different = !error_flag[2];
      size_t n = error_flag[3];
      if (TF_PREDICT_FALSE(!valid || !increasing || !different)) {
        auto* error_indices =
            (int64*) cpu_allocator()->AllocateRaw(sizeof(size_t),
                                                  sizeof(int64) * num_dims);
        d.memcpyDeviceToHost(error_indices,
                             indices_shaped.matrix<int64>().data() + n * num_dims,
                             sizeof(int64) * num_dims);
        TF_OP_REQUIRES_CUDA_SUCCESS(c, gpuEventRecord(copy_done, d.stream()));
        TF_OP_REQUIRES_CUDA_SUCCESS(c, gpuEventSynchronize(copy_done));

        string index = strings::StrCat("indices[", n, "] = [");
        for (int di = 0; di < num_dims; ++di) {
          strings::StrAppend(&index, error_indices[di], di < num_dims - 1 ? "," : "]");
        }
        cpu_allocator()->DeallocateRaw(error_indices);

        OP_REQUIRES_OK(c, [&]() -> Status {
          if (!valid) {
            return errors::InvalidArgument(
                index,
                " is out of bounds: need 0 <= index < [",
                str_util::Join(output_tensor_shape.dim_sizes(), ","), "]");
          }
          if (!increasing) {
            return errors::InvalidArgument(
                index,
                " is out of order. Many sparse ops require sorted indices.\n"
                "    Use `tf.sparse.reorder` to create a correctly ordered copy."
                "\n\n");
          }
          if (!different) {
            return errors::InvalidArgument(index, " is repeated");
          }
          return Status::OK();
        }());
      }
      cpu_allocator()->DeallocateRaw(error_flag);
      d.deallocate(d_error_flag);
      d.deallocate(d_order_ptr);
      gpuEventDestroy(copy_done);
    }

    std::vector<int64> strides(num_dims);
    if (num_dims > 0) {
      strides[num_dims - 1] = 1;
    }
    for (int i = num_dims - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * output_tensor_shape.dim_size(i + 1);
    }
    auto* d_strides_ptr = reinterpret_cast<int64*>(d.allocate(dim_vec_bytes));
    d.memcpyHostToDevice(d_strides_ptr, strides.data(), dim_vec_bytes);

    Eigen::Sizes<1> single;
    auto output_flat = output->flat<T>();
    output_flat.device(d) =
        default_value.scalar<T>().reshape(single).broadcast(output_flat.dimensions());

    auto* d_error_flag = reinterpret_cast<int*>(d.allocate(sizeof(int)));
    d.memset(d_error_flag, 0, sizeof(int));
    GpuLaunchConfig cfg = GetGpuLaunchConfig(num_elems, d);
    TF_CHECK_OK(GpuLaunchKernel(SparseToDenseKernel<T>, cfg.block_count,
                                cfg.thread_per_block, 0, d.stream(),
                                indices_shaped.matrix<int64>().data(),
                                sparse_values_b.vec<T>().data(),
                                output_flat.data(), num_elems,
                                d_strides_ptr, d_dim_ptr, num_dims, d_error_flag));

    gpuEvent_t copy_done;
    gpuEventCreate(&copy_done);
    auto* error_flag = (int*) cpu_allocator()->AllocateRaw(sizeof(size_t), sizeof(int));
    d.memcpyDeviceToHost(error_flag, d_error_flag, sizeof(int));
    TF_OP_REQUIRES_CUDA_SUCCESS(c, gpuEventRecord(copy_done, d.stream()));
    TF_OP_REQUIRES_CUDA_SUCCESS(c, gpuEventSynchronize(copy_done));

    bool kernel_error = error_flag[0];
    cpu_allocator()->DeallocateRaw(error_flag);
    d.deallocate(d_error_flag);
    d.deallocate(d_strides_ptr);
    d.deallocate(d_dim_ptr);
    gpuEventDestroy(copy_done);

    OP_REQUIRES(c, !kernel_error,
                errors::InvalidArgument(
                    "Indices are not valid (out of bounds).  Shape: ",
                    output->shape().DebugString()));
  }
};

}  // namespace functor

#define DEFINE_GPU_FUNCTOR(T)                                          \
  template struct functor::SparseToDenseFunctor<GPUDevice, T, int32>;  \
  template struct functor::SparseToDenseFunctor<GPUDevice, T, int64>;

TF_CALL_REAL_NUMBER_TYPES(DEFINE_GPU_FUNCTOR);
DEFINE_GPU_FUNCTOR(bool);
#undef DEFINE_GPU_FUNCTOR
}  // namespace tensorflow

#endif
