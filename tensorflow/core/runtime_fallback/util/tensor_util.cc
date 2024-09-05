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
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

using tfrt::DType;
using tfrt::Expected;
using tfrt::HostBuffer;
using tfrt::RCReference;
using tfrt::StringHostTensor;

// Moves one ref on HostBuffer to tensorflow::Tensor.
tensorflow::Tensor MoveHostBufferToTfTensor(RCReference<HostBuffer> host_buffer,
                                            DType dtype,
                                            const tfrt::TensorShape& shape) {
  llvm::SmallVector<tfrt::Index, 4> dims;
  shape.GetDimensions(&dims);

  auto deallocator = [](void* data, size_t len, void* arg) {
    auto* host_buffer = reinterpret_cast<HostBuffer*>(arg);
    host_buffer->DropRef();
  };

  CheckBoolCompatibility();
  // Transfer one HostBuffer ref to TFTensor.
  auto* data = host_buffer->data();
  auto size = host_buffer->size();
  OwnedTFTensor tf_tensor{TF_NewTensor(
      static_cast<TF_DataType>(GetTfDataType(dtype)), dims.data(), dims.size(),
      data, size, deallocator, host_buffer.release())};
  return TensorFromInterface(tf_tensor->tensor);
}

tensorflow::Tensor CopyShtToTfTensor(const StringHostTensor& sht) {
  llvm::SmallVector<tfrt::Index, 4> dims;
  sht.shape().GetDimensions(&dims);

  tensorflow::Tensor tensor(
      tensorflow::DT_STRING,
      tensorflow::TensorShape(
          llvm::SmallVector<int64_t, 4>(dims.begin(), dims.end())));

  auto len = tensor.NumElements();
  auto from = sht.strings();
  auto to = tensor.flat<tensorflow::tstring>();

  // TODO(tfrt-devs): Consider a more efficient way to pass string
  // tensors between TFRT and TF.
  for (int i = 0; i < len; ++i) {
    to(i).assign(from[i].data(), from[i].size());
  }
  return tensor;
}

}  // namespace tfd
}  // namespace tensorflow
