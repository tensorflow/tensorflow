/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/jit/xla_host_recv_device_context.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace tensorflow {

void XlaHostRecvDeviceContext::CopyDeviceTensorToCPU(
    const Tensor* device_tensor, StringPiece tensor_name, Device* device,
    Tensor* cpu_tensor, StatusCallback done) {
  DataType dtype = EncodePrimitiveTypeAsDataType(shape_.element_type()).value();
  TensorShape tensor_shape;
  Status status = XLAShapeToTensorShape(shape_, &tensor_shape);
  if (!status.ok()) {
    done(status);
    return;
  }

  *cpu_tensor = Tensor(dtype, tensor_shape);

  stream_->ThenMemcpy(cpu_tensor->data(), device_memory_base_,
                      device_memory_base_.size());
  stream_->ThenRecordEvent(&done_event_.get());
  if (auto st = stream_->BlockHostUntilDone(); !st.ok()) {
    done_event_.SetError(absl::InternalError(absl::StrFormat(
        "failed to synchronize send operation with a stream: %s",
        st.ToString())));
    return;
  }

  done_event_.SetStateConcrete();
  done(OkStatus());
}

}  // namespace tensorflow
