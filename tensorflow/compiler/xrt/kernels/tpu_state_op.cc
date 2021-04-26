/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Classes for allocating XLA literals in device memory and managing handles
// that refer to them.

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xrt/kernels/xrt_state_ops.h"
#include "tensorflow/compiler/xrt/xrt_tpu_device.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
REGISTER_KERNEL_BUILDER(Name("XRTAllocate")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("allocation")
                            .HostMemory("handle"),
                        XRTAllocateOp<XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTAllocateUninitialized")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("handle"),
                        XRTAllocateUninitializedOp<XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTAllocateFromTensor")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("inputs")
                            .HostMemory("handle"),
                        XRTAllocateFromTensorOp<XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTSubTuple")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<false, XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTSubTupleAndRelease")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<true, XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTMakeTuple")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("tuple_description")
                            .HostMemory("input_handles")
                            .HostMemory("output_handle"),
                        XRTMakeTupleOp<XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReadLiteral")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<false, XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTWriteLiteral")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("handle")
                            .HostMemory("literal")
                            .HostMemory("output_handle"),
                        XRTWriteLiteralOp<XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReadLiteralAndRelease")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<true, XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReadToTensor")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("handles")
                            .HostMemory("tensors"),
                        XRTReadToTensorOp<XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReleaseAllocationHandle")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("handle"),
                        XRTReleaseAllocationOp<XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(
    Name("XRTReleaseAllAllocations").Device(DEVICE_TPU_NODE),
    XRTReleaseAllAllocationsOp<XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTCompactAllocations").Device(DEVICE_TPU_NODE),
                        XRTCompactAllocationsOp<XRTTpuDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTMemoryInfo").Device(DEVICE_TPU_NODE),
                        XRTMemoryInfoOp<XRTTpuDeviceAccessor>);

}  // namespace tensorflow
