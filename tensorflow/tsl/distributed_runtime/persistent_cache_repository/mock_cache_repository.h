/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PERSISTENT_CACHE_REPOSITORY_MOCK_CACHE_REPOSITORY_H_
#define TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PERSISTENT_CACHE_REPOSITORY_MOCK_CACHE_REPOSITORY_H_

#include <string>

#include "tensorflow/tsl/distributed_runtime/persistent_cache_repository/persistent_cache_repository.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {

class MockCacheRepository : public PersistentCacheRepository {
 public:
  MockCacheRepository() : PersistentCacheRepository(Options{}) {}
  ~MockCacheRepository() override = default;

  MOCK_METHOD(absl::Status, Put,
              (const std::string& key, const std::string& serialized_entry),
              (override));
  MOCK_METHOD(absl::StatusOr<std::string>, Get, (const std::string& key),
              (override));
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PERSISTENT_CACHE_REPOSITORY_MOCK_CACHE_REPOSITORY_H_
