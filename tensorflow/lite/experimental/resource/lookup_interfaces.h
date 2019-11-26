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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_LOOKUP_INTERFACES_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_LOOKUP_INTERFACES_H_

#include <unordered_map>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/resource/lookup_util.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace resource {

/// WARNING: Experimental interface, subject to change.
// A resource hash table interface. It's similar to TensorFlow core's
// LookupInterface class. But it's identified with int32 ID in TFLite (instead
// of using Resource handle like TensorFlow).
class LookupInterface : public ResourceBase {
 public:
  virtual TfLiteStatus Lookup(TfLiteContext* context, const TfLiteTensor* keys,
                              TfLiteTensor* values,
                              const TfLiteTensor* default_value) = 0;
  virtual TfLiteStatus Import(TfLiteContext* context, const TfLiteTensor* keys,
                              const TfLiteTensor* values) = 0;
  virtual size_t Size() = 0;

  virtual TfLiteType GetKeyType() const = 0;
  virtual TfLiteType GetValueType() const = 0;
  virtual TfLiteStatus CheckKeyAndValueTypes(TfLiteContext* context,
                                             const TfLiteTensor* keys,
                                             const TfLiteTensor* values) = 0;
};

// Creates an resource hash table, shared among all the subgraphs with the
// given resource id if there is an existing one.
// WARNING: Experimental interface, subject to change.
void CreateHashtableResourceIfNotAvailable(ResourceMap* resources,
                                           int resource_id,
                                           TfLiteType key_dtype,
                                           TfLiteType value_dtype);

// Returns the corresponding resource hash table, or nullptr if none.
// WARNING: Experimental interface, subject to change.
LookupInterface* GetHashtableResource(ResourceMap* resources, int resource_id);

}  // namespace resource
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_LOOKUP_INTERFACES_H_
