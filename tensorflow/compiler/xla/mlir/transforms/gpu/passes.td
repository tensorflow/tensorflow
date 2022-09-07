/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_GPU_PASSES
#define XLA_GPU_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMemrefGetGlobalToArgPass :
    Pass<"xla-memref-get-global-to-arg", "mlir::ModuleOp"> {
  let summary = "Converts memref.get_global corresponding to lmhlo constants";

  let description = [{
    Replaces `memref.get_global` operations corresponding to the lmhlo constant
    arguments (arguments marked with `lmhlo.constant_name` attribute) to use
    the constant arguments directly.

    Once we used global constants for constant folding, we no longer need to
    keep them in the module, because they'll be in the binary constant section
    on the host, and we need them on the device.
  }];

  let constructor = "createConvertMemrefGetGlobalToArgPass()";

  let options = [
    Option<"min_num_elements_", "min-num-elements", "int64_t", /*default=*/"0",
           "Do not convert `memref.get_global` operation if the number of "
           "elements is smaller than the given value.">,
  ];
}

#endif  // XLA_GPU_PASSES
