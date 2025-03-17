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

// A minimal library exposing the naming logic used in tensor_bundle.
//
// A tensor bundle contains a metadata file and sharded data files, which all
// share a common pathname prefix.
//
// Given the prefix, the actual pathnames of the files can be queried via:
//
//   MetaFilename(prefix): pathname of the metadata file.
//   DataFilename(prefix, shard_id, num_shards): pathname of a data file.
//
// Typical usage includes forming a filepattern to match files on disk:
//
//   // To find the unique metadata file.
//   const string metadata_file = MetaFilename("/fs/train/ckpt-step");
//   Env::Default()->GetMatchingFiles(metadata_file, &path);
//
// Regexp can also be used: e.g. R"<prefix>.data-\d{5}-of-\d{5}" for data files.

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_NAMING_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_NAMING_H_

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

string MetaFilename(absl::string_view prefix);
string DataFilename(absl::string_view prefix, int32_t shard_id,
                    int32_t num_shards);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_NAMING_H_
