/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/autotune_result_wrapper.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

AutotuneResults ThreeAutotuneEntries(int32_t version) {
  AutotuneResults results;
  results.set_version(version);
  auto r1 = results.add_results();
  r1->set_device("dev1");
  r1->set_hlo("hlo1");
  r1->mutable_result()->set_scratch_bytes(1);

  auto r2 = results.add_results();
  r2->set_device("dev2");
  r2->set_hlo("hlo2");
  r2->mutable_result()->set_scratch_bytes(2);

  auto r3 = results.add_results();
  r3->set_device("dev3");
  r3->set_hlo("hlo3");
  r3->mutable_result()->set_scratch_bytes(3);

  return results;
}

TEST(AutotuneResultWrapperTest, FullRoundTrip) {
  std::vector<AutotuneResultWrapper> wrappers =
      AutotuneResultWrapper::AutotuneResultsToWrappers(
          ThreeAutotuneEntries(/*version=*/42));

  std::vector<std::pair<AutotuneResultWrapper::OpaqueKey,
                        AutotuneResultWrapper::OpaqueValue>>
      key_value_pairs;
  for (const auto& wrapper : wrappers) {
    key_value_pairs.push_back(std::make_pair(wrapper.Key(), wrapper.Value()));
  }

  std::vector<AutotuneResultWrapper> new_wrappers;
  for (const auto& [key, value] : key_value_pairs) {
    TF_ASSERT_OK_AND_ASSIGN(AutotuneResultWrapper wrapper,
                            AutotuneResultWrapper::FromKeyAndValue(key, value));
    new_wrappers.push_back(std::move(wrapper));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      AutotuneResults round_tripped,
      AutotuneResultWrapper::AutotuneResultsFromWrappers(new_wrappers));
  EXPECT_EQ(round_tripped.results_size(), 3);
  EXPECT_EQ(round_tripped.version(), 42);
  EXPECT_EQ(round_tripped.results(0).device(), "dev1");
  EXPECT_EQ(round_tripped.results(0).hlo(), "hlo1");
  EXPECT_EQ(round_tripped.results(0).result().scratch_bytes(), 1);
  EXPECT_EQ(round_tripped.results(1).device(), "dev2");
  EXPECT_EQ(round_tripped.results(1).hlo(), "hlo2");
  EXPECT_EQ(round_tripped.results(1).result().scratch_bytes(), 2);
  EXPECT_EQ(round_tripped.results(2).device(), "dev3");
  EXPECT_EQ(round_tripped.results(2).hlo(), "hlo3");
  EXPECT_EQ(round_tripped.results(2).result().scratch_bytes(), 3);
}

TEST(AutotuneResultWrapperTest, InconsistentVersions) {
  std::vector<AutotuneResultWrapper> wrappers =
      AutotuneResultWrapper::AutotuneResultsToWrappers(
          ThreeAutotuneEntries(/*version=*/42));
  auto inconsistent_wrappers = AutotuneResultWrapper::AutotuneResultsToWrappers(
      ThreeAutotuneEntries(/*version=*/43));
  wrappers.insert(wrappers.end(), inconsistent_wrappers.begin(),
                  inconsistent_wrappers.end());

  std::vector<std::pair<AutotuneResultWrapper::OpaqueKey,
                        AutotuneResultWrapper::OpaqueValue>>
      key_value_pairs;
  for (const auto& wrapper : wrappers) {
    key_value_pairs.push_back(std::make_pair(wrapper.Key(), wrapper.Value()));
  }

  std::vector<AutotuneResultWrapper> decoded_wrappers;
  for (const auto& [key, value] : key_value_pairs) {
    TF_ASSERT_OK_AND_ASSIGN(AutotuneResultWrapper wrapper,
                            AutotuneResultWrapper::FromKeyAndValue(key, value));
    decoded_wrappers.push_back(std::move(wrapper));
  }

  EXPECT_IS_NOT_OK(
      AutotuneResultWrapper::AutotuneResultsFromWrappers(decoded_wrappers));
}

}  // namespace
}  // namespace xla
