/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This include can't be in the conv_ops_fused_impl.h headers. See b/62899350.
#if GOOGLE_CUDA
#include "tensorflow/core/protobuf/autotuning.pb.h"
#endif  // GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_fused_impl.h"

namespace tensorflow {

#if GOOGLE_CUDA

namespace functor {
DECLARE_FUNCTOR_GPU_SPEC(Eigen::half);
}  // namespace functor

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
