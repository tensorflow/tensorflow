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

// For Google-internal use only.
//
// Supports serializing the autotune maps to string
// (SerializeAutotuneMaps), as well as deserializing them from
// string and injecting them into TF runtime
// (LoadSerializedAutotuneMaps).
//
// Aims to speed up the warmup time of neural nets.

#ifndef TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_AUTOTUNE_SERIALIZE_H_
#define TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_AUTOTUNE_SERIALIZE_H_

#include <string>

#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// TODO(b/189530096) Support autotune maps for more ops.
// Loads autotune maps from string output by SerializeAutotuneMaps and uses
// them to update the runtime autotune maps.
Status LoadSerializedAutotuneMaps(absl::string_view s);

// Serializes all the autotune maps into a string that can be decoded by
// LoadSerializedAutotuneMaps.
Status SerializeAutotuneMaps(std::string* output);

// Resets all autotune maps. For test use only.
void ResetAutotuneMaps();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_AUTOTUNE_SERIALIZE_H_
