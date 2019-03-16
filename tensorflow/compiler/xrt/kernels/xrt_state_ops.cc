/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>

#include "tensorflow/compiler/xrt/kernels/xrt_state_ops.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/local_client.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("XRTAllocate")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("allocation")
                            .HostMemory("handle"),
                        XRTAllocateOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTAllocate")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("allocation")
                            .HostMemory("handle"),
                        XRTAllocateOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTAllocateFromTensor")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("inputs")
                            .HostMemory("handle"),
                        XRTAllocateFromTensorOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTAllocateFromTensor")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("inputs")
                            .HostMemory("handle"),
                        XRTAllocateFromTensorOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTSubTuple")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<false, XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTSubTuple")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<false, XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTSubTupleAndRelease")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<true, XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTSubTupleAndRelease")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("base_handle")
                            .HostMemory("shape_index")
                            .HostMemory("output_handle"),
                        XRTSubTupleOp<true, XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTMakeTuple")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("tuple_description")
                            .HostMemory("input_handles")
                            .HostMemory("output_handle"),
                        XRTMakeTupleOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTMakeTuple")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("tuple_description")
                            .HostMemory("input_handles")
                            .HostMemory("output_handle"),
                        XRTMakeTupleOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReadLiteral")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<false, XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReadLiteral")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<false, XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTWriteLiteral")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle")
                            .HostMemory("literal")
                            .HostMemory("output_handle"),
                        XRTWriteLiteralOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTWriteLiteral")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle")
                            .HostMemory("literal")
                            .HostMemory("output_handle"),
                        XRTWriteLiteralOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReadLiteralAndRelease")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<true, XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReadLiteralAndRelease")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle")
                            .HostMemory("literal"),
                        XRTReadLiteralOp<true, XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReadToTensor")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handles")
                            .HostMemory("tensors"),
                        XRTReadToTensorOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReadToTensor")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handles")
                            .HostMemory("tensors"),
                        XRTReadToTensorOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReleaseAllocationHandle")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle"),
                        XRTReleaseAllocationOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReleaseAllocationHandle")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle"),
                        XRTReleaseAllocationOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTReleaseAllAllocations").Device(DEVICE_XLA_GPU),
                        XRTReleaseAllAllocationsOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTReleaseAllAllocations").Device(DEVICE_XLA_CPU),
                        XRTReleaseAllAllocationsOp<XRTGenericDeviceAccessor>);

}  // namespace tensorflow
