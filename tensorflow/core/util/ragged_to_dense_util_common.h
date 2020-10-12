/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_COMMON_H_
#define TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_COMMON_H_

#include <string>
#include <vector>

namespace tensorflow {
enum class RowPartitionType {
  FIRST_DIM_SIZE,
  VALUE_ROWIDS,
  ROW_LENGTHS,
  ROW_SPLITS,
  ROW_LIMITS,
  ROW_STARTS
};

std::string RowPartitionTypeToString(RowPartitionType row_partition_type);

std::vector<RowPartitionType> GetRowPartitionTypesHelper(
    const std::vector<std::string>& row_partition_type_strings);

int GetRaggedRank(const std::vector<RowPartitionType>& row_partition_types);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_COMMON_H_
