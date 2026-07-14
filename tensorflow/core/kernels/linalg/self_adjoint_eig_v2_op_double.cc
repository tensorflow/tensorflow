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

#include "tensorflow/core/kernels/linalg/self_adjoint_eig_v2_op_impl.h"

namespace tensorflow {

REGISTER_LINALG_OP("SelfAdjointEigV2", (SelfAdjointEigV2Op<double>), double);

// Deprecated kernel.
REGISTER_LINALG_OP("BatchSelfAdjointEigV2", (SelfAdjointEigV2Op<double>),
                   double);

}  // namespace tensorflow
