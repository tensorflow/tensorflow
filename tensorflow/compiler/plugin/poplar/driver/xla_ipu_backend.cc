/* Copyright 2017 Graphcore Ltd
 */

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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def.pb.h"

namespace tensorflow {

static bool OpFilter(KernelDef* kdef) {

  if (kdef->op() == "Angle") return false;
  if (kdef->op() == "Complex") return false;
  if (kdef->op() == "ComplexAbs") return false;
  if (kdef->op() == "Conj") return false;
  if (kdef->op() == "FFT") return false;
  if (kdef->op() == "FFT2D") return false;
  if (kdef->op() == "FFT3D") return false;
  if (kdef->op() == "IFFT") return false;
  if (kdef->op() == "IFFT2D") return false;
  if (kdef->op() == "IFFT3D") return false;
  if (kdef->op() == "Imag") return false;
  if (kdef->op() == "MaxPoolGradGrad") return false;
  if (kdef->op() == "Real") return false;

  return true;
}

REGISTER_XLA_BACKEND(DEVICE_IPU_XLA_JIT, kIpuAllTypes, OpFilter);

}  // namespace tensorflow
