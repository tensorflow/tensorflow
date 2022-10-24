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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_COPY_TO_DEVICE_NODE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_COPY_TO_DEVICE_NODE_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"

namespace tensorflow {

class CopyToDeviceNode : public EagerNode {
 public:
  CopyToDeviceNode(TensorHandle* src, TensorHandle* dst, Device* dstd,
                   const EagerContext& ctx, bool async, bool mirror)
      : EagerNode(),
        src_(src),
        dst_(dst),
        dstd_(dstd),
        ctx_(ctx),
        async_(async),
        mirror_(mirror) {
    if (async_) {
      src_->Ref();
      dst_->Ref();
    }
  }

  ~CopyToDeviceNode() override {
    if (async_) {
      src_->Unref();
      dst_->Unref();
    }
  }

  Status Run() override {
    tensorflow::Tensor tensor;
    profiler::ScopedMemoryDebugAnnotation op_annotation(
        "eager::CopyToDeviceNode", "dynamic", tensor.dtype(),
        [&tensor]() { return tensor.shape().DebugString(); });
    TF_RETURN_IF_ERROR(src_->CopyToDevice(ctx_, dstd_, &tensor));
    if (!async_ && mirror_) {
      Status s = dst_->AddLocalMirror(std::move(tensor), dstd_);
      // If a mirror was added since we called HasLocalMirror then just return
      // and ignore the error.
      if (s.ok() || (s.code() == error::Code::ALREADY_EXISTS)) {
        return OkStatus();
      }
      return s;
    } else {
      return dst_->SetTensor(std::move(tensor), dstd_);
    }
  }

  void Abort(Status status) override { dst_->Poison(status, dstd_); }

  string DebugString() const override {
    string out = "[CopyToDeviceNode]";
    strings::StrAppend(&out, " src_tensor: ", src_->DebugString());
    strings::StrAppend(&out, ", dst_tensor: ", dst_->DebugString());
    strings::StrAppend(&out, ", dst_device: ", dstd_ ? dstd_->name() : "[]");
    return out;
  }

  TensorHandle* dst() { return dst_; }

 private:
  TensorHandle* src_;
  TensorHandle* dst_;
  Device* dstd_;
  const EagerContext& ctx_;
  bool async_;
  bool mirror_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_COPY_TO_DEVICE_NODE_H_
