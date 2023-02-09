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

// We define the dialect here so that both structs and ops can refer to it.

#ifndef LHLO_GPU_OPS_BASE
#define LHLO_GPU_OPS_BASE

include "mlir/IR/OpBase.td"

def LmhloGpuDialect : Dialect {
  let name = "lmhlo_gpu";
  let cppNamespace = "::mlir::lmhlo_gpu";

  let useDefaultAttributePrinterParser = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // LHLO_GPU_OPS_BASE
