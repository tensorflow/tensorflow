/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/autotuner/directory_store.h"

#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::SizeIs;

autotuner::AutotuneEntry MakeEntry(std::string device,
                                   std::string explicit_version,
                                   std::string hlo_fingerprint,
                                   std::string codegen_version,
                                   std::string codegen_options_fingerprint,
                                   autotuner::Backend backend) {
  autotuner::AutotuneEntry entry;
  entry.mutable_key()->mutable_target()->set_device(device);
  entry.mutable_key()->mutable_target()->set_explicit_version(explicit_version);
  entry.mutable_key()->mutable_target()->set_hlo_fingerprint(hlo_fingerprint);
  entry.mutable_key()->mutable_environment()->set_codegen_version(
      codegen_version);
  entry.mutable_key()->mutable_environment()->set_codegen_options_fingerprint(
      codegen_options_fingerprint);
  entry.mutable_value()->mutable_optimal_config()->set_backend(backend);
  return entry;
}

class DirectoryStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        auto temp,
        tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
    temp_dir_ =
        std::make_unique<tsl::testing::TemporaryDirectory>(std::move(temp));
    cache_dir_ = temp_dir_->path();
  }

  void TearDown() override { temp_dir_.reset(); }

  std::unique_ptr<tsl::testing::TemporaryDirectory> temp_dir_;
  std::string cache_dir_;
};

TEST_F(DirectoryStoreTest, FileRoundTrip) {
  DirectoryStore store(cache_dir_, CacheMode::kReadWrite);

  autotuner::AutotuneTargetKey target_key;
  target_key.set_device("gpu");
  target_key.set_explicit_version("v1.0");
  target_key.set_hlo_fingerprint("fp1");

  // Read empty.
  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries1,
                       store.Read(target_key));
  EXPECT_THAT(entries1, SizeIs(0));

  // Write and Read.
  autotuner::AutotuneEntry entry1 = MakeEntry(
      "gpu", "v1.0", "fp1", "cg1", "opt1", autotuner::Backend::TRITON);
  EXPECT_OK(store.Write(entry1));

  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries2,
                       store.Read(target_key));
  ASSERT_THAT(entries2, SizeIs(1));
  EXPECT_EQ(entries2[0].value().optimal_config().backend(),
            autotuner::Backend::TRITON);
}

TEST_F(DirectoryStoreTest, EmptyExplicitVersionLayout) {
  DirectoryStore store(cache_dir_, CacheMode::kReadWrite);

  autotuner::AutotuneTargetKey target_key;
  target_key.set_device("gpu");
  target_key.set_explicit_version("");  // empty
  target_key.set_hlo_fingerprint("fp1");

  autotuner::AutotuneEntry entry1 =
      MakeEntry("gpu", "", "fp1", "cg1", "opt1", autotuner::Backend::TRITON);
  EXPECT_OK(store.Write(entry1));

  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries,
                       store.Read(target_key));
  ASSERT_THAT(entries, SizeIs(1));
}

TEST_F(DirectoryStoreTest, ReadAllReturnsUnimplemented) {
  DirectoryStore store(cache_dir_, CacheMode::kReadWrite);
  EXPECT_THAT(store.ReadAll(), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(DirectoryStoreTest, PersistenceAcrossInstances) {
  {
    DirectoryStore store1(cache_dir_, CacheMode::kReadWrite);
    autotuner::AutotuneEntry entry1 = MakeEntry(
        "gpu", "v1.0", "fp1", "cg1", "opt1", autotuner::Backend::TRITON);
    EXPECT_OK(store1.Write(entry1));
  }

  // Create fresh instance pointing to the same folder.
  DirectoryStore store2(cache_dir_, CacheMode::kReadOnly);
  autotuner::AutotuneTargetKey target_key;
  target_key.set_device("gpu");
  target_key.set_explicit_version("v1.0");
  target_key.set_hlo_fingerprint("fp1");

  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries,
                       store2.Read(target_key));
  ASSERT_THAT(entries, SizeIs(1));
}

TEST_F(DirectoryStoreTest, WriteDifferentEnvironmentOverwrites) {
  DirectoryStore store(cache_dir_, CacheMode::kReadWrite);

  autotuner::AutotuneTargetKey target_key;
  target_key.set_device("gpu");
  target_key.set_explicit_version("v1.0");
  target_key.set_hlo_fingerprint("fp1");

  autotuner::AutotuneEntry entry1 = MakeEntry(
      "gpu", "v1.0", "fp1", "cg1", "opt1", autotuner::Backend::TRITON);
  EXPECT_OK(store.Write(entry1));

  // Write different environment to same target.
  autotuner::AutotuneEntry entry2 = MakeEntry(
      "gpu", "v1.0", "fp1", "cg2", "opt2", autotuner::Backend::CUBLASLT);
  EXPECT_OK(store.Write(entry2));

  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries,
                       store.Read(target_key));
  // Should only hold 1 entry (the latest one) because we overwrite on
  // same target key.
  ASSERT_THAT(entries, SizeIs(1));
  EXPECT_EQ(entries[0].key().environment().codegen_version(), "cg2");
  EXPECT_EQ(entries[0].value().optimal_config().backend(),
            autotuner::Backend::CUBLASLT);
}

TEST_F(DirectoryStoreTest, ReadOnlyWriteIsIgnored) {
  {
    DirectoryStore store(cache_dir_, CacheMode::kReadOnly);
    autotuner::AutotuneEntry entry = MakeEntry(
        "gpu", "v1.0", "fp1", "cg1", "opt1", autotuner::Backend::TRITON);
    EXPECT_OK(store.Write(entry));
  }

  // Verify it was not written to disk.
  DirectoryStore store_rw(cache_dir_, CacheMode::kReadWrite);
  autotuner::AutotuneTargetKey target_key;
  target_key.set_device("gpu");
  target_key.set_explicit_version("v1.0");
  target_key.set_hlo_fingerprint("fp1");

  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries,
                       store_rw.Read(target_key));
  EXPECT_THAT(entries, SizeIs(0));
}

TEST_F(DirectoryStoreTest, ReadInvalidProtoFails) {
  DirectoryStore store(cache_dir_, CacheMode::kReadWrite);
  autotuner::AutotuneEntry entry = MakeEntry(
      "gpu", "v1.0", "fp1", "cg1", "opt1", autotuner::Backend::TRITON);

  // Write valid entry first to ensure directories are created.
  EXPECT_OK(store.Write(entry));

  // Overwrite the file with invalid content.
  // The file path should be <cache_dir_>/gpu/v1.0/fp1.pb
  std::string file_path = cache_dir_ + "/gpu/v1.0/fp1.pb";

  std::ofstream ofs(file_path, std::ios::trunc);
  ofs << "invalid garbage proto content";
  ofs.close();

  autotuner::AutotuneTargetKey target_key;
  target_key.set_device("gpu");
  target_key.set_explicit_version("v1.0");
  target_key.set_hlo_fingerprint("fp1");

  EXPECT_THAT(store.Read(target_key),
              StatusIs(absl::StatusCode::kInternal,
                       testing::HasSubstr("Failed to parse cache entry")));
}

}  // namespace
}  // namespace xla
