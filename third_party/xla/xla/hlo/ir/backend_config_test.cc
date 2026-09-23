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

#include "xla/hlo/ir/backend_config.h"

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "google/protobuf/struct.pb.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

const int kNumThreads = 100;
const int kNumRepetitions = 100;

// This string has to be in a canonical form (without spaces and new lines)
// since the == operator does not canonicalize the raw strings before comparing
// them.
constexpr absl::string_view kRawString =
    R"({"operation_queue_id":"0","fusion_backend_config":{"kind":"__triton_gemm","triton_gemm_config":{"block_m":"256","block_n":"256","block_k":"32","split_k":"1","num_stages":"1","num_warps":"16","num_ctas":"1","is_tma_allowed":false,"is_warp_specialization_allowed":false,"waves_per_eu":"0","group_size":"0"}},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"})";

template <typename Input, typename CheckFn>
void RunThreaded(Input input, CheckFn check_fn) {
  for (int i = 0; i < kNumRepetitions; ++i) {
    BackendConfigWrapper source(input);

    absl::Notification all_threads_created;
    std::vector<std::unique_ptr<std::thread>> threads;

    for (int i = 0; i < kNumThreads; ++i) {
      threads.emplace_back(std::make_unique<std::thread>([&] {
        all_threads_created.WaitForNotification();
        check_fn(source);
      }));
    }
    all_threads_created.Notify();

    for (int i = 0; i < kNumThreads; ++i) {
      threads[i]->join();
    }
  }
}

TEST(BackendConfigWrapperTest, ConcurrentGetProto) {
  RunThreaded(std::string{kRawString}, [](BackendConfigWrapper& source) {
    gpu::GpuBackendConfig proto;
    TF_EXPECT_OK(source.GetProto(&proto));
    EXPECT_TRUE(proto.has_fusion_backend_config());
    BackendConfigWrapper wrapped(proto);
    EXPECT_TRUE(wrapped == source);
  });
}

TEST(BackendConfigWrapperTest, ConcurrentGetRawString) {
  BackendConfigWrapper source_json(std::string{kRawString});
  gpu::GpuBackendConfig proto;
  TF_EXPECT_OK(source_json.GetProto(&proto));

  RunThreaded(proto, [](BackendConfigWrapper& source) {
    std::string raw_string = source.GetRawString();
    EXPECT_EQ(raw_string, kRawString);
    BackendConfigWrapper wrapped(raw_string);
    EXPECT_TRUE(wrapped == source);
  });
}

TEST(BackendConfigWrapperTest, AssignmentToNonEmptyIsOK) {
  BackendConfigWrapper a(std::string{kRawString});
  BackendConfigWrapper b(std::string{kRawString});
  a = std::move(b);
  EXPECT_TRUE(a == BackendConfigWrapper(std::string{kRawString}));
}

TEST(BackendConfigWrapperTest, AssignmentDoesNotDeadlock) {
  BackendConfigWrapper source;
  BackendConfigWrapper& ref = source;
  source = std::move(ref);
}

TEST(BackendConfigWrapperTest, SelfComparisonDoesNotDeadlock) {
  BackendConfigWrapper source(std::string{kRawString});
  EXPECT_TRUE(source == source);
}

TEST(BackendConfigWrapperTest, ComparisonDoesNotDeadlock) {
  BackendConfigWrapper source_json(std::string{kRawString});
  gpu::GpuBackendConfig proto;
  TF_EXPECT_OK(source_json.GetProto(&proto));
  RunThreaded(std::string{kRawString}, [&proto](BackendConfigWrapper& source) {
    BackendConfigWrapper other_first(proto);
    EXPECT_TRUE(other_first == source);
    BackendConfigWrapper other_second(proto);
    EXPECT_TRUE(source == other_second);
  });
}

TEST(BackendConfigRawStringCacheTest, EqualProtosShareOneEntry) {
  OpMetadata metadata;
  metadata.set_op_name("shared");
  OpMetadata other_metadata;
  other_metadata.set_op_name("other");

  BackendConfigRawStringCache cache;
  BackendConfigWrapper first(metadata);
  BackendConfigWrapper second(metadata);
  BackendConfigWrapper other(other_metadata);
  BackendConfigWrapper from_string(std::string{kRawString});
  first.GetRawString(&cache);
  second.GetRawString(&cache);
  other.GetRawString(&cache);
  from_string.GetRawString(&cache);

  EXPECT_EQ(cache.size(), 2);
  EXPECT_EQ(first.GetRawString(),
            BackendConfigWrapper(metadata).GetRawString());
  EXPECT_EQ(second.GetRawString(), first.GetRawString());
  EXPECT_EQ(other.GetRawString(),
            BackendConfigWrapper(other_metadata).GetRawString());
  EXPECT_NE(other.GetRawString(), first.GetRawString());
  EXPECT_EQ(from_string.GetRawString(), kRawString);
}

TEST(BackendConfigRawStringCacheTest, EqualBytesOfDifferentTypesDoNotShare) {
  // Both messages serialize field 1 as the same string, so only the type
  // tells them apart.
  OpMetadata metadata;
  metadata.set_op_type("x");
  Payload payload;
  payload.set_value("x");
  ASSERT_EQ(metadata.SerializeAsString(), payload.SerializeAsString());

  BackendConfigRawStringCache cache;
  BackendConfigWrapper metadata_wrapper(metadata);
  BackendConfigWrapper payload_wrapper(payload);
  metadata_wrapper.GetRawString(&cache);
  payload_wrapper.GetRawString(&cache);

  EXPECT_EQ(cache.size(), 2);
  EXPECT_EQ(metadata_wrapper.GetRawString(),
            BackendConfigWrapper(metadata).GetRawString());
  EXPECT_EQ(payload_wrapper.GetRawString(),
            BackendConfigWrapper(payload).GetRawString());
  EXPECT_NE(metadata_wrapper.GetRawString(), payload_wrapper.GetRawString());
}

// GpuBackendConfig reaches AlgorithmProto.tuning_knobs, a map field. A map
// parsed from JSON prints in the text order of its entries, so equal maps in
// different text orders must never share a string.
TEST(BackendConfigRawStringCacheTest, TypesWithMapFieldsBypassTheCache) {
  constexpr absl::string_view kKnobsAscending =
      R"({"cudnn_conv_backend_config":{"algorithm":{"tuning_knobs":{"1":"10","2":"20"}}}})";
  constexpr absl::string_view kKnobsDescending =
      R"({"cudnn_conv_backend_config":{"algorithm":{"tuning_knobs":{"2":"20","1":"10"}}}})";
  auto parsed = [](absl::string_view raw_string) {
    auto wrapper =
        std::make_unique<BackendConfigWrapper>(std::string{raw_string});
    CHECK_OK(wrapper->ApplyFnOnProto<gpu::GpuBackendConfig>(
        [](gpu::GpuBackendConfig*) { return absl::OkStatus(); }));
    return wrapper;
  };
  gpu::GpuBackendConfig proto;
  ASSERT_OK(BackendConfigWrapper(std::string{kRawString}).GetProto(&proto));

  BackendConfigRawStringCache cache;
  BackendConfigWrapper first(proto);
  BackendConfigWrapper second(proto);
  std::unique_ptr<BackendConfigWrapper> ascending = parsed(kKnobsAscending);
  std::unique_ptr<BackendConfigWrapper> descending = parsed(kKnobsDescending);
  first.GetRawString(&cache);
  second.GetRawString(&cache);
  ascending->GetRawString(&cache);
  descending->GetRawString(&cache);

  EXPECT_EQ(cache.size(), 0);
  EXPECT_EQ(first.GetRawString(), kRawString);
  EXPECT_EQ(second.GetRawString(), kRawString);
  EXPECT_EQ(ascending->GetRawString(), parsed(kKnobsAscending)->GetRawString());
  EXPECT_EQ(descending->GetRawString(),
            parsed(kKnobsDescending)->GetRawString());
}

// GetRawString encodes through the cache only for a wrapper that holds a
// proto and no string yet; wrappers that are empty, string backed, already
// encoded without a cache, parsed by GetProto, copied or moved leave the cache
// alone, and a mutation makes the next call encode again.
TEST(BackendConfigRawStringCacheTest, GetRawStringWithCacheEncodesOnce) {
  OpMetadata metadata;
  metadata.set_op_name("shared");
  const std::string expected = BackendConfigWrapper(metadata).GetRawString();
  auto shared_entries = [](const BackendConfigWrapper& wrapper) {
    BackendConfigRawStringCache cache;
    wrapper.GetRawString(&cache);
    return cache.size();
  };
  BackendConfigWrapper without_cache(metadata);
  EXPECT_EQ(without_cache.GetRawString(nullptr), expected);
  EXPECT_EQ(shared_entries(without_cache), 0);

  BackendConfigWrapper empty;
  EXPECT_EQ(shared_entries(empty), 0);
  EXPECT_EQ(empty.GetRawString(), "");

  BackendConfigWrapper from_string(std::string{kRawString});
  EXPECT_EQ(shared_entries(from_string), 0);
  gpu::GpuBackendConfig parsed;
  ASSERT_OK(from_string.GetProto(&parsed));
  EXPECT_EQ(shared_entries(from_string), 0);
  EXPECT_EQ(from_string.GetRawString(), kRawString);

  BackendConfigWrapper from_proto(metadata);
  EXPECT_EQ(shared_entries(from_proto), 1);
  EXPECT_EQ(shared_entries(from_proto), 0);
  EXPECT_EQ(from_proto.GetRawString(), expected);

  BackendConfigWrapper computed_by_get(metadata);
  EXPECT_EQ(computed_by_get.GetRawString(), expected);
  EXPECT_EQ(shared_entries(computed_by_get), 0);

  ASSERT_OK(from_proto.ApplyFnOnProto<OpMetadata>([](OpMetadata* proto) {
    proto->set_op_type("mutated");
    return absl::OkStatus();
  }));
  EXPECT_EQ(shared_entries(from_proto), 1);
  OpMetadata mutated = metadata;
  mutated.set_op_type("mutated");
  EXPECT_EQ(from_proto.GetRawString(),
            BackendConfigWrapper(mutated).GetRawString());

  BackendConfigWrapper source(metadata);
  BackendConfigWrapper copy(source);
  EXPECT_EQ(shared_entries(copy), 1);
  EXPECT_EQ(shared_entries(source), 1);
  EXPECT_EQ(copy.GetRawString(), expected);

  BackendConfigWrapper moved_from(metadata);
  BackendConfigWrapper moved_to(std::string{kRawString});
  moved_to = std::move(moved_from);
  EXPECT_EQ(shared_entries(moved_to), 1);
  EXPECT_EQ(moved_to.GetRawString(), expected);
  EXPECT_EQ(shared_entries(moved_from), 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(moved_from.GetRawString(), "");
}

// Threads fill, read and mutate one wrapper at the same time. GetRawString's
// reference is not stable across a concurrent mutation, so the threads read
// through GetProto and the string is checked once the threads are done.
TEST(BackendConfigRawStringCacheTest, ConcurrentGetRawStringAndMutation) {
  BackendConfigWrapper source_json(std::string{kRawString});
  gpu::GpuBackendConfig proto;
  ASSERT_OK(source_json.GetProto(&proto));
  const std::string expected_bytes = proto.SerializeAsString();

  for (int repetition = 0; repetition < kNumRepetitions; ++repetition) {
    BackendConfigWrapper source(proto);
    absl::Notification all_threads_created;
    std::vector<std::unique_ptr<std::thread>> threads;
    for (int i = 0; i < kNumThreads; ++i) {
      threads.emplace_back(std::make_unique<std::thread>([&, i] {
        all_threads_created.WaitForNotification();
        BackendConfigRawStringCache cache;
        switch (i % 3) {
          case 0:
            source.GetRawString(&cache);
            break;
          case 1: {
            gpu::GpuBackendConfig seen;
            EXPECT_OK(source.GetProto(&seen));
            EXPECT_EQ(seen.SerializeAsString(), expected_bytes);
            break;
          }
          default:
            EXPECT_OK(source.ApplyFnOnProto<gpu::GpuBackendConfig>(
                [](gpu::GpuBackendConfig*) { return absl::OkStatus(); }));
            source.GetRawString(&cache);
            break;
        }
      }));
    }
    all_threads_created.Notify();
    for (auto& thread : threads) {
      thread->join();
    }
    BackendConfigRawStringCache cache;
    source.GetRawString(&cache);
    EXPECT_EQ(source.GetRawString(), kRawString);
  }
}

// google.protobuf.Value reaches Struct (a map field) and ListValue, which
// points back to Value. Whatever the order in which the types are queried, a
// type that reaches the map through the cycle is never cached, and answering
// one query must not memoize a wrong answer for a type inside the cycle.
TEST(BackendConfigRawStringCacheTest, CyclesDoNotPoisonTheMemoization) {
  google::protobuf::Value value;
  value.set_number_value(1.0);
  google::protobuf::ListValue list;
  *list.add_values() = value;
  google::protobuf::Struct dictionary;
  (*dictionary.mutable_fields())["k"] = value;
  for (const std::vector<const tsl::protobuf::Message*>& order :
       {std::vector<const tsl::protobuf::Message*>{&value, &list, &dictionary},
        std::vector<const tsl::protobuf::Message*>{&list, &value, &dictionary},
        std::vector<const tsl::protobuf::Message*>{&dictionary, &list,
                                                   &value}}) {
    BackendConfigRawStringCache cache;
    for (const tsl::protobuf::Message* proto : order) {
      BackendConfigWrapper first(*proto);
      BackendConfigWrapper second(*proto);
      EXPECT_EQ(first.GetRawString(&cache),
                BackendConfigWrapper(*proto).GetRawString());
      EXPECT_EQ(second.GetRawString(&cache), first.GetRawString());
    }
    EXPECT_EQ(cache.size(), 0);
  }
}

TEST(BackendConfigRawStringCacheTest, TypesWithExtensionRangesBypassTheCache) {
  tsl::protobuf::MessageOptions options;
  options.set_deprecated(true);

  BackendConfigRawStringCache cache;
  BackendConfigWrapper first(options);
  BackendConfigWrapper second(options);
  first.GetRawString(&cache);
  second.GetRawString(&cache);

  EXPECT_EQ(cache.size(), 0);
  EXPECT_EQ(first.GetRawString(), BackendConfigWrapper(options).GetRawString());
  EXPECT_EQ(second.GetRawString(), first.GetRawString());
}

}  // namespace
}  // namespace xla
