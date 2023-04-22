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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/proto/descriptor_pool_registry.h"

namespace tensorflow {
namespace {

struct LocalDescriptorPool {
  static Status Function(
      tensorflow::protobuf::DescriptorPool const** desc_pool,
      std::unique_ptr<tensorflow::protobuf::DescriptorPool>* owned_desc_pool) {
    *desc_pool = ::tensorflow::protobuf::DescriptorPool::generated_pool();
    if (*desc_pool == nullptr) {
      return errors::InvalidArgument("Problem loading protobuf generated_pool");
    }
    return Status::OK();
  }
};

REGISTER_DESCRIPTOR_POOL("", LocalDescriptorPool::Function);
REGISTER_DESCRIPTOR_POOL("local://", LocalDescriptorPool::Function);

}  // namespace
}  // namespace tensorflow
