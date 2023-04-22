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
#include <unordered_map>
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

inline std::string RowPartitionTypeToString(
    RowPartitionType row_partition_type) {
  switch (row_partition_type) {
    case RowPartitionType::FIRST_DIM_SIZE:
      return "FIRST_DIM_SIZE";
    case RowPartitionType::VALUE_ROWIDS:
      return "VALUE_ROWIDS";
    case RowPartitionType::ROW_LENGTHS:
      return "ROW_LENGTHS";
    case RowPartitionType::ROW_SPLITS:
      return "ROW_SPLITS";
    case RowPartitionType::ROW_LIMITS:
      return "ROW_LIMITS";
    case RowPartitionType::ROW_STARTS:
      return "ROW_STARTS";
    default:
      return "UNKNOWN ROW PARTITION TYPE";
  }
}

inline std::vector<RowPartitionType> GetRowPartitionTypesHelper(
    const std::vector<std::string>& row_partition_type_strings) {
  static const auto kStringToType =
      new std::unordered_map<std::string, RowPartitionType>(
          {{"FIRST_DIM_SIZE", RowPartitionType::FIRST_DIM_SIZE},
           {"VALUE_ROWIDS", RowPartitionType::VALUE_ROWIDS},
           {"ROW_LENGTHS", RowPartitionType::ROW_LENGTHS},
           {"ROW_SPLITS", RowPartitionType::ROW_SPLITS},
           {"ROW_LIMITS", RowPartitionType::ROW_LIMITS},
           {"ROW_STARTS", RowPartitionType::ROW_STARTS}});
  std::vector<RowPartitionType> result;
  for (const auto& type_str : row_partition_type_strings) {
    const auto iter = kStringToType->find(type_str);
    if (iter == kStringToType->end()) {
      break;
    }
    result.push_back(iter->second);
  }
  return result;
}

inline int GetRaggedRank(
    const std::vector<RowPartitionType>& row_partition_types) {
  if (row_partition_types.empty()) {
    return 0;
  }
  if (row_partition_types[0] == RowPartitionType::FIRST_DIM_SIZE) {
    return row_partition_types.size() - 1;
  }
  return row_partition_types.size();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_RAGGED_TO_DENSE_UTIL_COMMON_H_
