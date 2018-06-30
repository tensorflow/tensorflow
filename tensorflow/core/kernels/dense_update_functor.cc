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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/dense_update_functor.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <>
struct DenseUpdate<CPUDevice, string, ASSIGN> {
  void operator()(const CPUDevice& d, typename TTypes<string>::Flat params,
                  typename TTypes<string>::ConstFlat update) {
    if (params.dimension(0) == 1) {
      params.data()->resize(update.data()->size());
      auto work = [&params, &update](int64 start, int64 end) {
        memmove(const_cast<char*>(params.data()->data()) + start,
                update.data()->data() + start, end - start);
      };
      d.parallelFor(update.data()->size(),
                    Eigen::TensorOpCost(.1,  // chosen to force large chunks
                                        .1, 0),
                    work);
    } else {
      auto work = [&params, &update](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          params.data()[i].resize(update.data()[i].size());
          memmove(const_cast<char*>(params.data()[i].data()),
                  update.data()[i].data(), update.data()[i].size());
        }
      };
      int64 estimated_string_size;
      if (update.size() > 0) {
        // first element of the tensor seems as good a guess as any of the sizes
        // of the strings contained within...
        estimated_string_size =
            std::max(update.data()[0].size(), sizeof(string));
      } else {
        estimated_string_size = sizeof(string);
      }
      d.parallelFor(
          params.dimension(0),
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          work);
    }
  }
};

}  // namespace functor

#define CPU_DENSE_COPY(T)                                                \
  case DataTypeToEnum<T>::value: {                                       \
    functor::DenseUpdate<CPUDevice, T, ASSIGN> copy_functor_;            \
    copy_functor_(context->eigen_device<CPUDevice>(), tensor->flat<T>(), \
                  from.flat<T>());                                       \
    break;                                                               \
  }

#define INSTANTIATE_GET_VARIANT_COPY_FN(DEVICE, TYPE_CALLER, TYPE_DENSE_COPY) \
  template <>                                                                 \
  Status VariantCopyFn<DEVICE>(OpKernelContext * context, const Tensor& from, \
                               Tensor* to) {                                  \
    PersistentTensor tmp;                                                     \
    Tensor* tensor;                                                           \
    AllocatorAttributes attr;                                                 \
    attr.set_gpu_compatible(true);                                            \
    attr.set_nic_compatible(true);                                            \
    TF_RETURN_IF_ERROR(context->allocate_persistent(                          \
        from.dtype(), from.shape(), &tmp, &tensor, attr));                    \
    switch (from.dtype()) {                                                   \
      TYPE_CALLER(TYPE_DENSE_COPY);                                           \
      default:                                                                \
        return errors::InvalidArgument(                                       \
            "VariantCopyFn: Could not perform a deep copy of variant "        \
            "element of type: ",                                              \
            DataTypeString(from.dtype()),                                     \
            " using device: ", context->device()->name());                    \
    }                                                                         \
    *to = *tensor;                                                            \
    return Status::OK();                                                      \
  }

INSTANTIATE_GET_VARIANT_COPY_FN(CPUDevice, TF_CALL_ALL_TYPES, CPU_DENSE_COPY);

#if GOOGLE_CUDA
#define GPU_DENSE_COPY(T)                                                \
  case DataTypeToEnum<T>::value: {                                       \
    functor::DenseUpdate<GPUDevice, T, ASSIGN> copy_functor_;            \
    copy_functor_(context->eigen_device<GPUDevice>(), tensor->flat<T>(), \
                  from.flat<T>());                                       \
    break;                                                               \
  }
#define TF_CALL_GPU_AND_ADDITIONAL_TYPES(T) \
  TF_CALL_GPU_ALL_TYPES(T);                 \
  TF_CALL_int32(T);                         \
  TF_CALL_int64(T);
INSTANTIATE_GET_VARIANT_COPY_FN(GPUDevice, TF_CALL_GPU_AND_ADDITIONAL_TYPES,
                                GPU_DENSE_COPY);
#undef TF_CALL_GPU_AND_ADDITIONAL_TYPES
#undef GPU_DENSE_COPY
#endif  // GOOGLE_CUDA

#undef CPU_DENSE_COPY
#undef INSTANTIATE_GET_VARIANT_COPY_FN

}  // namespace tensorflow
