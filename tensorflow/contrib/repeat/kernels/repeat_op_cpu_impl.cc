/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow{

template <typename T>
void RepeatCPUImplSingleThreaded(const Tensor& input,
                   const typename TTypes<int32>::ConstFlat& repeats_flat,
                   int axis, Tensor* output) {
  auto input_flat = input.flat<T>();
  auto output_flat = output->flat<T>();
    
  // a batch is inner axes > axis
  size_t batch_size = 1;
  int32 dims = input.shape().dims();
  for (int32 i = axis + 1; i < dims; ++i) {
    batch_size *= input.shape().dim_size(i);
  }
  int64 num_batch = input_flat.size() / batch_size;
  
  const T* in = input_flat.data();
  T* out = output_flat.data();
  
  // copy an in_batch to its out_batches
  auto handle_batch = [&in, batch_size, &out](int32 repeat) {
    for (int64 j = 0; j < repeat; ++j) {
      std::copy(in, in + batch_size, out);
      out += batch_size;
    }
    in += batch_size;
    return;
  };
  
  if (repeats_flat.size() == 1) {
    for (int64 i = 0; i < num_batch; ++i) {
      handle_batch(repeats_flat(0));
    }
  } else {
    for (int64 i = 0; i < num_batch; ++i) {
      handle_batch(repeats_flat(i % repeats_flat.size()));
    }
  }
}

template <typename T>
void RepeatCPUImplMultiThreaded(DeviceBase* d, const Tensor& input,
                     const typename TTypes<int32>::ConstFlat& repeats_flat,
                     int axis, int64 cost_per_unit, Tensor* output) {
  auto input_flat = input.flat<T>();
  auto output_flat = output->flat<T>();
  
  // a batch is inner axes > axis
  // a group is inner axes >= axis
  int64 batch_size = 1;
  int32 dims = input.shape().dims();
  for (int32 i = axis + 1; i < dims; ++i) {
    batch_size *= input.shape().dim_size(i);
  }
  int64 group_pre_size = batch_size * input.shape().dim_size(axis);
  int64 group_size = batch_size * output->shape().dim_size(axis);
  
  auto worker_threads = d->tensorflow_cpu_worker_threads();
  int num_threads = std::min(4, worker_threads->num_threads);
  // strings define a different amount of work (generally much more) compared
  // with standard POD, so we parallelize differently.
  if (!std::is_same<T, string>::value) {
    num_threads =
        static_cast<int>(std::min<int64>(num_threads, output_flat.size() / 4096));
  }
  
  if (num_threads == 0) {
    RepeatCPUImplSingleThreaded<T>(input, repeats_flat, axis, output);
  }
  
  auto work = [input_flat, repeats_flat, axis,
               batch_size, group_pre_size, group_size, &output_flat](
      int64 out_begin_index, int64 out_end_index) {
    const T* in = input_flat.data();
    T* out = output_flat.data();
    T* out_start = out + out_begin_index;
    T* out_end = out + out_end_index;
    
    // handle partial group at start
    int64 skip_group = out_begin_index / group_size;
    in += skip_group * group_pre_size;
    out += skip_group * group_size;
    
    if (out_begin_index % group_size != 0) {
      for (int64 j = 0; j < repeats_flat.size(); ++j) {
        for (int64 k = 0; k < repeats_flat(j); ++k) {
          if (out + batch_size <= out_start) {
            out += batch_size;
            continue;
          }
          
          int64 offset = out_start - out;
          offset = offset>0 ? offset : 0;
          if (out + batch_size > out_end) {
            std::copy(in + offset, in + (out_end-out), out + offset);
            return;
          }
          std::copy(in + offset, in + batch_size, out + offset);
          
          out += batch_size;
        }
        in += batch_size;
      }
    }
    
    // handle remaining data
    int64 group_to_cpy = (out_end-out) / group_size + 1;
    for (int64 i = 0; i < group_to_cpy; ++i) {
      for (int64 j = 0; j < repeats_flat.size(); ++j) {
        for (int64 k = 0; k < repeats_flat(j); ++k) {
          if (out + batch_size > out_end) {
            std::copy(in, in + (out_end-out), out);
            return;
          }
          std::copy(in, in + batch_size, out);
          out += batch_size;
        }
        in += batch_size;
      }
    }
  };

  Shard(worker_threads->num_threads, worker_threads->workers, output_flat.size(),
        cost_per_unit, work);
}

#define REGISTER(T)                                          \
template void RepeatCPUImplSingleThreaded<T>(                              \
    const Tensor& input,                                     \
    const typename TTypes<int32>::ConstFlat& repeats_flat,   \
    int axis, Tensor* output);

#define REGISTER_V2(T)                                       \
template void RepeatCPUImplMultiThreaded<T>(                            \
    DeviceBase* d, const Tensor& input,                      \
    const typename TTypes<int32>::ConstFlat& repeats_flat,   \
    int axis, int64 cost_per_unit, Tensor* output);
    
TF_CALL_ALL_TYPES(REGISTER);
TF_CALL_ALL_TYPES(REGISTER_V2);

#undef REGISTER
#undef REGISTER_V2

} // end namespace tensorflow
