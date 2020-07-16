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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/tile_functor_cpu.h"

namespace tensorflow {
namespace functor {

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;

#define DEFINE_TYPE(T)                        \
  template struct Tile<SYCLDevice, T, int32>; \
  template struct Tile<SYCLDevice, T, int64>;

TF_CALL_bool(DEFINE_TYPE);
TF_CALL_float(DEFINE_TYPE);
TF_CALL_bfloat16(DEFINE_TYPE);
TF_CALL_double(DEFINE_TYPE);
TF_CALL_uint8(DEFINE_TYPE);
TF_CALL_int32(DEFINE_TYPE);
TF_CALL_int16(DEFINE_TYPE);
TF_CALL_int64(DEFINE_TYPE);

#undef DEFINE_TYPE
#endif  // TENSORFLOW_USE_SYCL

}  // end namespace functor
}  // end namespace tensorflow
