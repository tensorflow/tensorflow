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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_STATIC_HASHTABLE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_STATIC_HASHTABLE_H_

#include <unordered_map>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/resource/lookup_interfaces.h"
#include "tensorflow/lite/experimental/resource/lookup_util.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace resource {
namespace internal {

// A static hash table class. This hash table allows initialization one time in
// its life cycle. This hash table implements Tensorflow core's HashTableV2 op.
template <typename KeyType, typename ValueType>
class StaticHashtable : public tflite::resource::LookupInterface {
 public:
  explicit StaticHashtable(TfLiteType key_type, TfLiteType value_type)
      : key_type_(key_type), value_type_(value_type) {}
  ~StaticHashtable() override {}

  // Finds the corresponding value of the given keys tensor in the map and
  // copies the result data to the given values tensor. If there is no matching
  // value, it will write the default value into the matched position instead.
  TfLiteStatus Lookup(TfLiteContext* context, const TfLiteTensor* keys,
                      TfLiteTensor* values,
                      const TfLiteTensor* default_value) override;

  // Inserts the given key and value tensor data into the hash table.
  TfLiteStatus Import(TfLiteContext* context, const TfLiteTensor* keys,
                      const TfLiteTensor* values) override;

  // Returns the item size of the hash table.
  size_t Size() override { return map_.size(); }

  TfLiteType GetKeyType() const override { return key_type_; }
  TfLiteType GetValueType() const override { return value_type_; }

  TfLiteStatus CheckKeyAndValueTypes(TfLiteContext* context,
                                     const TfLiteTensor* keys,
                                     const TfLiteTensor* values) override {
    TF_LITE_ENSURE_EQ(context, keys->type, key_type_);
    TF_LITE_ENSURE_EQ(context, values->type, value_type_);
    return kTfLiteOk;
  }

  // Returns true if the hash table is initialized.
  bool IsInitialized() override { return is_initialized_; }

  size_t GetMemoryUsage() override { return map_.size() * sizeof(ValueType); }

 private:
  TfLiteType key_type_;
  TfLiteType value_type_;

  std::unordered_map<KeyType, ValueType> map_;
  bool is_initialized_ = false;
};

::tflite::resource::LookupInterface* CreateStaticHashtable(
    TfLiteType key_type, TfLiteType value_type);

}  // namespace internal

}  // namespace resource
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_STATIC_HASHTABLE_H_
