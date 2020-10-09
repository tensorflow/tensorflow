/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/tile_functor_cpu.h"

namespace tensorflow {
namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;

template struct Tile<CPUDevice, Variant, int32>;
template struct Tile<CPUDevice, Variant, int64>;

}  // end namespace functor
}  // end namespace tensorflow
