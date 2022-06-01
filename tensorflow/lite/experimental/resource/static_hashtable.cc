/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/resource/static_hashtable.h"

#include <memory>
#include "tensorflow/lite/experimental/resource/lookup_interfaces.h"

namespace tflite {
namespace resource {
namespace internal {

template <typename KeyType, typename ValueType>
TfLiteStatus StaticHashtable<KeyType, ValueType>::Lookup(
    TfLiteContext* context, const TfLiteTensor* keys, TfLiteTensor* values,
    const TfLiteTensor* default_value) {
  if (!is_initialized_) {
    TF_LITE_KERNEL_LOG(context,
                       "hashtable need to be initialized before using");
    return kTfLiteError;
  }
  const int size =
      MatchingFlatSize(GetTensorShape(keys), GetTensorShape(values));

  auto key_tensor_reader = TensorReader<KeyType>(keys);
  auto value_tensor_writer = TensorWriter<ValueType>(values);
  auto default_value_tensor_reader = TensorReader<ValueType>(default_value);
  ValueType first_default_value = default_value_tensor_reader.GetData(0);

  for (int i = 0; i < size; ++i) {
    auto result = map_.find(key_tensor_reader.GetData(i));
    if (result != map_.end()) {
      value_tensor_writer.SetData(i, result->second);
    } else {
      value_tensor_writer.SetData(i, first_default_value);
    }
  }

  // This is for a string tensor case in order to write buffer back to the
  // actual tensor destination. Otherwise, it does nothing since the scalar data
  // will be written into the tensor storage directly.
  value_tensor_writer.Commit();

  return kTfLiteOk;
}

template <typename KeyType, typename ValueType>
TfLiteStatus StaticHashtable<KeyType, ValueType>::Import(
    TfLiteContext* context, const TfLiteTensor* keys,
    const TfLiteTensor* values) {
  // Import nodes can be invoked twice because the converter will not extract
  // the initializer graph separately from the original graph. The invocations
  // after the first call will be ignored.
  if (is_initialized_) {
    return kTfLiteOk;
  }

  const int size =
      MatchingFlatSize(GetTensorShape(keys), GetTensorShape(values));

  auto key_tensor_reader = TensorReader<KeyType>(keys);
  auto value_tensor_writer = TensorReader<ValueType>(values);
  for (int i = 0; i < size; ++i) {
    map_.insert({key_tensor_reader.GetData(i), value_tensor_writer.GetData(i)});
  }

  is_initialized_ = true;
  return kTfLiteOk;
}

LookupInterface* CreateStaticHashtable(TfLiteType key_type,
                                       TfLiteType value_type) {
  if (key_type == kTfLiteInt64 && value_type == kTfLiteString) {
    return new StaticHashtable<std::int64_t, std::string>(key_type, value_type);
  } else if (key_type == kTfLiteString && value_type == kTfLiteInt64) {
    return new StaticHashtable<std::string, std::int64_t>(key_type, value_type);
  }
  return nullptr;
}

}  // namespace internal

void CreateHashtableResourceIfNotAvailable(ResourceMap* resources,
                                           int resource_id,
                                           TfLiteType key_dtype,
                                           TfLiteType value_dtype) {
  if (resources->count(resource_id) != 0) {
    return;
  }
  auto* hashtable = internal::CreateStaticHashtable(key_dtype, value_dtype);
  resources->emplace(resource_id, std::unique_ptr<LookupInterface>(hashtable));
}

LookupInterface* GetHashtableResource(ResourceMap* resources, int resource_id) {
  auto it = resources->find(resource_id);
  if (it != resources->end()) {
    return static_cast<LookupInterface*>(it->second.get());
  }
  return nullptr;
}

}  // namespace resource
}  // namespace tflite
