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

#include "tensorflow/compiler/xrt/kernels/xrt_state_ops.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"

namespace tensorflow {
namespace {

class XRTMetricsCollectOp : public OpKernel {
 public:
  explicit XRTMetricsCollectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTMetricsCollectOp::Compute";

    const Tensor& metrics_proto = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(metrics_proto.shape()),
                errors::Internal("request input should be a string scalar"));
    xrt::XRTMetricsCollect metrics;
    OP_REQUIRES(ctx, metrics.ParseFromString(metrics_proto.scalar<tstring>()()),
                errors::InvalidArgument(
                    "Unable to parse request input to XRTMetricsCollect"));

    xla::StatusOr<xrt::MetricsReport> collected_metrics_or =
        CollectMetrics(metrics);
    OP_REQUIRES_OK(ctx, collected_metrics_or.status());
    xrt::MetricsReport collected_metrics =
        collected_metrics_or.ConsumeValueOrDie();
    Tensor output(DT_STRING, TensorShape({}));
    output.scalar<tstring>()() = collected_metrics.SerializeAsString();
    ctx->set_output(0, output);
  }
};

}  // namespace

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

REGISTER_KERNEL_BUILDER(Name("XRTAllocateUninitialized")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle"),
                        XRTAllocateUninitializedOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTAllocateUninitialized")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle"),
                        XRTAllocateUninitializedOp<XRTGenericDeviceAccessor>);

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

REGISTER_KERNEL_BUILDER(Name("XRTCompactAllocations").Device(DEVICE_XLA_GPU),
                        XRTCompactAllocationsOp<XRTGenericDeviceAccessor>);
REGISTER_KERNEL_BUILDER(Name("XRTCompactAllocations").Device(DEVICE_XLA_CPU),
                        XRTCompactAllocationsOp<XRTGenericDeviceAccessor>);

REGISTER_KERNEL_BUILDER(Name("XRTMetricsCollect").Device(DEVICE_CPU),
                        XRTMetricsCollectOp);

}  // namespace tensorflow
