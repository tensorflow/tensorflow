/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_compilation_cache_persistence.h"

#include <memory>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(XlaCompilationCachePersistenceTest, RegisterUnregister) {
  TF_ASSERT_OK(RegisterXlaCompilationCacheSaver([] { return nullptr; }));

  // Registration will fail unless the factory is unregistered.
  EXPECT_THAT(RegisterXlaCompilationCacheSaver([] { return nullptr; }),
              testing::StatusIs(error::INTERNAL));

  UnregisterXlaCompilationCacheSaver();

  TF_ASSERT_OK(RegisterXlaCompilationCacheSaver([] { return nullptr; }));
  UnregisterXlaCompilationCacheSaver();

  TF_ASSERT_OK(RegisterXlaCompilationCacheLoader([] { return nullptr; }));

  // Registration will fail unless the factory is unregistered.
  EXPECT_THAT(RegisterXlaCompilationCacheLoader([] { return nullptr; }),
              testing::StatusIs(error::INTERNAL));

  UnregisterXlaCompilationCacheLoader();

  TF_ASSERT_OK(RegisterXlaCompilationCacheLoader([] { return nullptr; }));
  UnregisterXlaCompilationCacheLoader();
}

TEST(XlaCompilationCachePersistenceTest, Create) {
  class MockSaver : public XlaCompilationCacheSaver {
   public:
    Status Save(const XlaSerializedCacheEntry& entry) override {
      return Status::OK();
    }
  };

  class MockLoader : public XlaCompilationCacheLoader {
   public:
    StatusOr<absl::optional<XlaSerializedCacheEntry>> TryLoad(
        const XlaSerializedCacheKey& key) override {
      XlaSerializedCacheEntry entry;
      return {entry};
    }
  };

  // Nothing registered should produce nullptr
  ASSERT_EQ(CreateXlaCompilationCacheSaver(), nullptr);
  ASSERT_EQ(CreateXlaCompilationCacheLoader(), nullptr);

  int num_save_create = 0;
  int num_load_create = 0;
  TF_ASSERT_OK(RegisterXlaCompilationCacheSaver([&num_save_create] {
    num_save_create++;
    return std::make_unique<MockSaver>();
  }));

  TF_ASSERT_OK(RegisterXlaCompilationCacheLoader([&num_load_create] {
    num_load_create++;
    return std::make_unique<MockLoader>();
  }));

  (void)CreateXlaCompilationCacheSaver();
  (void)CreateXlaCompilationCacheSaver();
  EXPECT_EQ(num_save_create, 2);

  (void)CreateXlaCompilationCacheLoader();
  (void)CreateXlaCompilationCacheLoader();
  (void)CreateXlaCompilationCacheLoader();
  EXPECT_EQ(num_load_create, 3);

  UnregisterXlaCompilationCacheSaver();
  UnregisterXlaCompilationCacheLoader();
}

}  // namespace
}  // namespace tensorflow
