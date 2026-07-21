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

#include "xla/backends/autotuner/in_memory_store.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::SizeIs;

autotuner::AutotuneEntry MakeEntry(const std::string& device,
                                   const std::string& explicit_version,
                                   const std::string& hlo_fingerprint,
                                   const std::string& codegen_version,
                                   const std::string& codegen_options_fp,
                                   autotuner::Backend backend) {
  autotuner::AutotuneEntry entry;
  autotuner::AutotuneTargetKey* target = entry.mutable_key()->mutable_target();
  target->set_device(device);
  target->set_explicit_version(explicit_version);
  target->set_hlo_fingerprint(hlo_fingerprint);
  autotuner::AutotuneEnvironmentKey* env =
      entry.mutable_key()->mutable_environment();
  env->set_codegen_version(codegen_version);
  env->set_codegen_options_fingerprint(codegen_options_fp);
  entry.mutable_value()->mutable_optimal_config()->set_backend(backend);
  return entry;
}

autotuner::AutotuneTargetKey MakeTargetKey(const std::string& device,
                                           const std::string& explicit_version,
                                           const std::string& hlo_fingerprint) {
  autotuner::AutotuneTargetKey key;
  key.set_device(device);
  key.set_explicit_version(explicit_version);
  key.set_hlo_fingerprint(hlo_fingerprint);
  return key;
}

class InMemoryStoreTest : public ::testing::Test {
 protected:
  void SetUp() override { InMemoryStore::Clear(); }
};

TEST_F(InMemoryStoreTest, WriteThenRead) {
  InMemoryStore store;

  autotuner::AutotuneEntry entry =
      MakeEntry("gpu", "v1", "fp1", "cg1", "opt1", autotuner::Backend::TRITON);
  EXPECT_THAT(store.Write(entry), IsOk());

  EXPECT_THAT(store.Read(MakeTargetKey("gpu", "v1", "fp1")),
              IsOkAndHolds(SizeIs(1)));
}

TEST_F(InMemoryStoreTest, ReadMissReturnsEmpty) {
  InMemoryStore store;
  EXPECT_THAT(store.Read(MakeTargetKey("gpu", "v1", "missing")),
              IsOkAndHolds(SizeIs(0)));
}

TEST_F(InMemoryStoreTest, ReadReturnsOnlyMatchingFingerprint) {
  InMemoryStore store;

  EXPECT_THAT(store.Write(MakeEntry("gpu", "v1", "fp1", "cg1", "opt1",
                                    autotuner::Backend::TRITON)),
              IsOk());
  EXPECT_THAT(store.Write(MakeEntry("gpu", "v1", "fp2", "cg1", "opt1",
                                    autotuner::Backend::TRITON)),
              IsOk());

  // Should only return the entry for "fp1".
  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries,
                       store.Read(MakeTargetKey("gpu", "v1", "fp1")));
  ASSERT_THAT(entries, SizeIs(1));
  EXPECT_EQ(entries[0].key().target().hlo_fingerprint(), "fp1");
}

TEST_F(InMemoryStoreTest, WriteOverwritesEntryWithSameTargetKey) {
  InMemoryStore store;

  EXPECT_THAT(store.Write(MakeEntry("gpu", "v1", "fp1", "cg1", "opt1",
                                    autotuner::Backend::TRITON)),
              IsOk());
  // Same target key, different value.
  EXPECT_THAT(store.Write(MakeEntry("gpu", "v1", "fp1", "cg1", "opt1",
                                    autotuner::Backend::CUBLASLT)),
              IsOk());

  ASSERT_OK_AND_ASSIGN(std::vector<autotuner::AutotuneEntry> entries,
                       store.Read(MakeTargetKey("gpu", "v1", "fp1")));
  ASSERT_THAT(entries, SizeIs(1));
  EXPECT_EQ(entries[0].value().optimal_config().backend(),
            autotuner::Backend::CUBLASLT);
}

TEST_F(InMemoryStoreTest, WriteDifferentEnvironmentAppends) {
  InMemoryStore store;

  EXPECT_THAT(store.Write(MakeEntry("gpu", "v1", "fp1", "cg1", "opt1",
                                    autotuner::Backend::TRITON)),
              IsOk());
  // Same target, different environment -> new slot.
  EXPECT_THAT(store.Write(MakeEntry("gpu", "v1", "fp1", "cg2", "opt2",
                                    autotuner::Backend::TRITON)),
              IsOk());

  EXPECT_THAT(store.Read(MakeTargetKey("gpu", "v1", "fp1")),
              IsOkAndHolds(SizeIs(2)));
}

TEST_F(InMemoryStoreTest, ReadAllFlattensAllTargets) {
  InMemoryStore store;

  EXPECT_THAT(store.Write(MakeEntry("gpu", "v1", "fp1", "cg1", "opt1",
                                    autotuner::Backend::TRITON)),
              IsOk());
  EXPECT_THAT(store.Write(MakeEntry("gpu", "v1", "fp2", "cg1", "opt1",
                                    autotuner::Backend::TRITON)),
              IsOk());

  EXPECT_THAT(store.ReadAll(), IsOkAndHolds(SizeIs(2)));
}

TEST_F(InMemoryStoreTest, GlobalStorageIsShared) {
  InMemoryStore store1;
  InMemoryStore store2;

  EXPECT_THAT(store1.Write(MakeEntry("gpu", "v1", "fp1", "cg1", "opt1",
                                     autotuner::Backend::TRITON)),
              IsOk());
  EXPECT_THAT(store2.Read(MakeTargetKey("gpu", "v1", "fp1")),
              IsOkAndHolds(SizeIs(1)));
}

}  // namespace
}  // namespace xla
