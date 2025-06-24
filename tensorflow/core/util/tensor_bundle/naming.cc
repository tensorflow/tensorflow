/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/tensor_bundle/naming.h"

#include <cstdint>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

string MetaFilename(absl::string_view prefix) {
  return strings::Printf("%.*s.index", static_cast<int>(prefix.size()),
                         prefix.data());
}

string DataFilename(absl::string_view prefix, int32_t shard_id,
                    int32_t num_shards) {
  DCHECK_GT(num_shards, 0);
  DCHECK_LT(shard_id, num_shards);
  return strings::Printf("%.*s.data-%05d-of-%05d",
                         static_cast<int>(prefix.size()), prefix.data(),
                         shard_id, num_shards);
}

}  // namespace tensorflow
