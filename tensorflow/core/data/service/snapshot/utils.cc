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
#include "tensorflow/core/data/service/snapshot/utils.h"

#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/byte_size.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace data {

ByteSize EstimatedSize(const std::vector<Tensor>& tensors) {
  ByteSize byte_size;
  for (const Tensor& tensor : tensors) {
    TensorProto proto;
    tensor.AsProtoTensorContent(&proto);
    byte_size += ByteSize::Bytes(proto.ByteSizeLong());
  }
  return byte_size;
}

}  // namespace data
}  // namespace tensorflow
