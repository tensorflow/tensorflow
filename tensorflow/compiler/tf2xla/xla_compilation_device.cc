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
    // Respects the aligment request because there is alignment checking even
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
  bool ShouldAllocateEmptyTensors() override { return true; }

  void GetStats(AllocatorStats* stats) override { stats->Clear(); }

 private:
  // Don't run any constructors or destructors for complex objects,
  // since there is no backing store for the tensor to run them
  // on. strings are the only complex objects currently stored in
  // Tensors. If others are added, this set of overrides must be
  // extended to include them.
  void RunStringCtor(string* p, size_t n) override {}
  void RunStringDtor(string* p, size_t n) override {}
  void RunResourceCtor(ResourceHandle* p, size_t n) override {}
  void RunResourceDtor(ResourceHandle* p, size_t n) override {}
};

XlaCompilationDevice::XlaCompilationDevice(const SessionOptions& options,
                                           DeviceType type)
    : LocalDevice(
          options,
          Device::BuildDeviceAttributes(
              strings::StrCat("/device:", type.type(), ":0"), type,
              Bytes(256 << 20), DeviceLocality(),
              strings::StrCat("device: XLA compilation device ", type.type()))),
      allocator_(new XlaCompilationAllocator()) {}

XlaCompilationDevice::~XlaCompilationDevice() {}

Allocator* XlaCompilationDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_.get();
}

void XlaCompilationDevice::Compute(OpKernel* op_kernel,
                                   OpKernelContext* context) {
  VLOG(4) << "XlaCompilationDevice::Compute "
          << SummarizeNodeDef(op_kernel->def());
  auto* b = XlaContext::Get(context).builder();
  xla::OpMetadata metadata;
  metadata.set_op_type(op_kernel->type_string());
  metadata.set_op_name(op_kernel->name());
  b->SetOpMetadata(metadata);

  auto sharding_parse_result = ParseShardingFromDevice(
      op_kernel->def(), std::numeric_limits<int>::max());
  OP_REQUIRES_OK(context, sharding_parse_result.status());
  absl::optional<xla::OpSharding> op_sharding =
      sharding_parse_result.ValueOrDie();

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

XlaExpression::XlaExpression() = default;

void XlaExpression::set_handle(const xla::XlaOp& h) { handle_ = h; }

void XlaExpression::set_constant_value(Tensor value) {
  has_constant_value_ = true;
  constant_value_ = std::move(value);
}

}  // namespace tensorflow
