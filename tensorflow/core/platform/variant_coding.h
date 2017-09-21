/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLATFORM_VARIANT_CODING_H_
#define TENSORFLOW_PLATFORM_VARIANT_CODING_H_

#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"

#ifdef PLATFORM_GOOGLE
#include "tensorflow/core/platform/google/variant_cord_coding.h"
#endif

namespace tensorflow {
namespace port {

// Encodes an array of Variant objects in to the given string.
// `variant_array` is assumed to point to an array of `n` Variant objects.
void EncodeVariantList(const Variant* variant_array, int64 n, string* out);

// Decodes an array of Variant objects from the given string.
// `variant_array` is assumed to point to an array of `n` Variant objects.
bool DecodeVariantList(const string& in, Variant* variant_array, int64 n);

}  // end namespace port
}  // end namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_VARIANT_CODING_H_
