/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_MOCK_RESOURCE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_MOCK_RESOURCE_H_

#include "tensorflow/lite/experimental/resource/resource_base.h"

namespace tflite {
namespace resource {

// Mock resource implementation for unit tests verifying type mismatch handling.
template <ResourceBase::ResourceType Type>
class MockTypedResource : public ResourceBase {
 public:
  ResourceType GetResourceType() const override { return Type; }
  bool IsInitialized() override { return true; }
};

using MockHashTableResource =
    MockTypedResource<ResourceBase::ResourceType::kHashTable>;
using MockVariableResource =
    MockTypedResource<ResourceBase::ResourceType::kResourceVariable>;

}  // namespace resource
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_MOCK_RESOURCE_H_
