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

#ifndef TENSORFLOW_CORE_UTIL_PROTO_DESCRIPTOR_POOL_REGISTRY_H_
#define TENSORFLOW_CORE_UTIL_PROTO_DESCRIPTOR_POOL_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

class DescriptorPoolRegistry {
 public:
  typedef std::function<Status(
      tensorflow::protobuf::DescriptorPool const** desc_pool,
      std::unique_ptr<tensorflow::protobuf::DescriptorPool>* owned_desc_pool)>
      DescriptorPoolFn;

  // Returns a pointer to a global DescriptorPoolRegistry object.
  static DescriptorPoolRegistry* Global();

  // Returns a pointer to a descriptor pool function for the given source.
  DescriptorPoolFn* Get(const string& source);

  // Registers a descriptor pool factory.
  void Register(const string& source, const DescriptorPoolFn& pool_fn);

 private:
  std::map<string, DescriptorPoolFn> fns_;
};

namespace descriptor_pool_registration {

class DescriptorPoolRegistration {
 public:
  DescriptorPoolRegistration(
      const string& source,
      const DescriptorPoolRegistry::DescriptorPoolFn& pool_fn) {
    DescriptorPoolRegistry::Global()->Register(source, pool_fn);
  }
};

}  // namespace descriptor_pool_registration

#define REGISTER_DESCRIPTOR_POOL(source, pool_fn) \
  REGISTER_DESCRIPTOR_POOL_UNIQ_HELPER(__COUNTER__, source, pool_fn)

#define REGISTER_DESCRIPTOR_POOL_UNIQ_HELPER(ctr, source, pool_fn) \
  REGISTER_DESCRIPTOR_POOL_UNIQ(ctr, source, pool_fn)

#define REGISTER_DESCRIPTOR_POOL_UNIQ(ctr, source, pool_fn)       \
  static descriptor_pool_registration::DescriptorPoolRegistration \
      descriptor_pool_registration_fn_##ctr(source, pool_fn)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_PROTO_DESCRIPTOR_POOL_REGISTRY_H_
