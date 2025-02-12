/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_LOOKUP_TABLE_INIT_OP_H_
#define TENSORFLOW_CORE_KERNELS_LOOKUP_TABLE_INIT_OP_H_

#include "tensorflow/core/kernels/initializable_lookup_table.h"

namespace tensorflow {
namespace lookup {

// Helper function to initialize an InitializableLookupTable from a text file.
absl::Status InitializeTableFromTextFile(const string& filename,
                                         int64_t vocab_size, char delimiter,
                                         int32_t key_index, int32_t value_index,
                                         Env* env,
                                         InitializableLookupTable* table);

}  // namespace lookup
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LOOKUP_TABLE_INIT_OP_H_
