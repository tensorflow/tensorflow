/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/convolution_thunk_internal.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "xla/tsl/framework/contraction/eigen_contraction_kernel.h"  // IWYU pragma: keep
#endif

CONV2D_INSTANTIATE_TEMPLATE(Eigen::DefaultDevice, float);
CONV2D_INSTANTIATE_TEMPLATE(Eigen::ThreadPoolDevice, float);

CONV3D_INSTANTIATE_TEMPLATE(Eigen::DefaultDevice, float);
CONV3D_INSTANTIATE_TEMPLATE(Eigen::ThreadPoolDevice, float);
