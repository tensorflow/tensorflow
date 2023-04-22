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

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"

#include <functional>
#include <memory>

#include "tensorflow/compiler/tf2xla/frontend_attributes_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

// The XlaCompilationAllocator doesn't actually back any Tensors with storage
// buffers of values: instead for each Tensor it stores a
// XlaExpression which corresponds to the XLA computation
// represented by the Tensor.
class XlaCompilationAllocator : public Allocator {
 public:
  XlaCompilationAllocator() {}
  ~XlaCompilationAllocator() override {}

  string Name() override { return "xla_compilation"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    // Regardless of the size requested, always allocates an XlaExpression.
    // Respects the alignment request because there is alignment checking even
    // for Tensors whose data is never accessed.
    void* p = port::AlignedMalloc(sizeof(XlaExpression), alignment);
    XlaExpression* expression = reinterpret_cast<XlaExpression*>(p);
    new (expression) XlaExpression();
    return expression;
  }

  void DeallocateRaw(void* ptr) override {
    XlaExpression* expression = reinterpret_cast<XlaExpression*>(ptr);
    expression->~XlaExpression();
    port::AlignedFree(ptr);
  }

  // Make sure that even tensors with 0 elements have allocated
  // buffers, so they get ids to track.
  //
  // NOTE: It is the caller's responsibility to track whether an allocated
  // object is a buffer or an opaque handle. In particular, when this allocator
  // is used, the caller must not run any constructors or destructors for
  // complex objects, since there is no backing store for the tensor in which to
  // place their outputs.
  bool AllocatesOpaqueHandle() const override { return true; }
};

XlaCompilationDevice::XlaCompilationDevice(const SessionOptions& options,
                                           DeviceType type)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               absl::StrCat("/device:", type.type(), ":0"),
                               type, Bytes(256 << 20), DeviceLocality(),
                               absl::StrCat("device: XLA compilation device ",
                                            type.type()))),
      allocator_(new XlaCompilationAllocator()) {}

XlaCompilationDevice::~XlaCompilationDevice() {}

Allocator* XlaCompilationDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_.get();
}

// Attaches location from the node stack trace to metadata. As a heuristic,
// picks the last frame which does not contain the "tensorflow/python" substring
// (making exception for frames containing "test" to allow for testing the
// feature).
static void AttachLocationToMetadata(xla::OpMetadata& metadata,
                                     OpKernel* op_kernel, XlaContext& context) {
  if (const AbstractStackTrace* stack_trace =
          context.StackTraceForNodeName(op_kernel->def().name())) {
    if (absl::optional<StackFrame> frame = stack_trace->LastUserFrame()) {
      metadata.set_source_file(frame->file_name);
      metadata.set_source_line(frame->line_number);
    }
  }
}

void XlaCompilationDevice::Compute(OpKernel* op_kernel,
                                   OpKernelContext* context) {
  VLOG(4) << "XlaCompilationDevice::Compute "
          << FormatNodeDefForError(op_kernel->def());
  XlaContext& xla_context = XlaContext::Get(context);
  auto* b = xla_context.builder();
  xla::OpMetadata metadata;
  metadata.set_op_type(op_kernel->type_string());
  metadata.set_op_name(op_kernel->name());
  AttachLocationToMetadata(metadata, op_kernel, xla_context);
  b->SetOpMetadata(metadata);

  auto sharding_parse_result =
      ParseShardingFromDevice(op_kernel->def(), std::numeric_limits<int>::max(),
                              /*add_metadata=*/false);
  OP_REQUIRES_OK(context, sharding_parse_result.status());
  absl::optional<xla::OpSharding> op_sharding =
      sharding_parse_result.ValueOrDie();

  auto frontend_attributes_result =
      GetFrontendAttributesFromAttrSlice(AttrSlice(op_kernel->def()));
  OP_REQUIRES_OK(context, frontend_attributes_result.status());
  absl::optional<xla::FrontendAttributes> attributes =
      frontend_attributes_result.ValueOrDie();

  xla::FrontendAttributes merged_attributes = b->frontend_attributes();
  if (attributes.has_value()) {
    merged_attributes.mutable_map()->insert(attributes.value().map().begin(),
                                            attributes.value().map().end());
  }
  xla::XlaScopedFrontendAttributesAssignment assign_frontend_attributes(
      b, std::move(merged_attributes));

  // If no sharding metadata is found, XLA is free to use whatever device it
  // wants. In practice this usually has the effect of placing things on device
  // 0.
  xla::XlaScopedShardingAssignment assign_sharding(b, op_sharding);
  op_kernel->Compute(context);

  b->ClearOpMetadata();
  VLOG(4) << "Done";
}

Status XlaCompilationDevice::Sync() { return Status::OK(); }

Status XlaCompilationDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  return errors::InvalidArgument(
      "XLACompilationDevice::MakeTensorFromProto should not be called");
}

}  // namespace tensorflow
