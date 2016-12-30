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

#include "tensorflow/core/kernels/cast_op_impl.h"

#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetCpuCastFromFloat(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, CPUDevice, float);
  if (dst_dtype == DT_BFLOAT16) {
    return [](OpKernelContext* ctx, const Tensor& inp, Tensor* out) {
      int64 N = out->NumElements();
      auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
      auto work = [&inp, &out](int64 start, int64 end) {
        FloatToBFloat16(inp.flat<float>().data() + start,
                        out->flat<bfloat16>().data() + start, end - start);
      };
      Shard(worker_threads->num_threads, worker_threads->workers, N, 2, work);
    };
  }

  return nullptr;
}

#if GOOGLE_CUDA
std::function<void(OpKernelContext*, const Tensor&, Tensor*)>
GetGpuCastFromFloat(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, GPUDevice, float);
  CAST_CASE(GPUDevice, float, bfloat16);
  return nullptr;
}
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
