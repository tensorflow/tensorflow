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

#ifdef TFRT_GPU_OPS
#else
#define TFRT_GPU_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/compiler/opdefs/tfrt_op_interfaces.td"
include "tfrt/compiler/opdefs/tfrt_traits.td"
include "tfrt_fallback.td"

def TFRT_GPU_Dialect : Dialect {
  let name = "gpurt";

  let description = [{
    The TFRT GPU Dialect.
  }];

  let cppNamespace = "::tfrt::gpu";
}

class Gpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<TFRT_GPU_Dialect, mnemonic, traits> {
}

// TODO(b/260267885): We may add a device argument when we want to support
// GPU MIG.
def TransferToDeviceOp: Gpu_Op<"transfer_to_device"> {
  let summary = "Transfer a CPU tensor to device.";

  let description = [{
    Transfer a CPU tensor to device.

    Example:
      %device_tensor = gpurt.transfer_to_device %cpu_tensor
  }];

  let arguments = (ins TFTensorType);
  let results = (outs TFTensorType);
  let assemblyFormat = "operands attr-dict";
}

// TODO(b/260267885): We may add a device argument when we want to support
// GPU MIG.
def TransferFromDeviceOp: Gpu_Op<"transfer_from_device"> {
  let summary = "Transfer a tensor from device.";

  let description = [{
    Transfer a tensor from device.

    Example:
      %cpu_tensor = gpurt.transfer_from_device %device_tensor
  }];

  let arguments = (ins TFTensorType);
  let results = (outs TFTensorType);
  let assemblyFormat = "operands attr-dict";
}

// TODO(b/260267885): We may add a device argument when we want to support
// GPU MIG.
def MaybeTransferVariableOp: Gpu_Op<"maybe_transfer_variable"> {
  let summary = "Transfer a CPU variable tensor to device.";
  let description = [{
    Transfer a CPU variable tensor to device if the variable has not been
    transferred before.

    Example:
      %device_var = gpurt.maybe_transfer_variable %cpu_var
  }];

  let arguments = (ins TFTensorType);
  let results = (outs TFTensorType);
  let assemblyFormat = "operands attr-dict";
}

#endif // TFRT_GPU_OPS
