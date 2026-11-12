/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/backends/cpu/runtime/rng_state_lib.h"
#include "xla/executable_run_options.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::xla::cpu::BufferAllocationInfo;

absl::int128 ReadInt128(const void* ptr) {
  uint64_t words[2];
  std::memcpy(words, ptr, sizeof(words));
  return absl::MakeInt128(static_cast<int64_t>(words[1]), words[0]);
}

class SharedRngFunction final : public XlaCompiledCpuFunction {
 public:
  SharedRngFunction() : XlaCompiledCpuFunction(GetStaticData()) {}

  absl::int128 out0() const { return ReadInt128(result_data(0)); }
  absl::int128 out1() const { return ReadInt128(result_data(1)); }

 private:
  static const StaticData& GetStaticData() {
    static const StaticData* const kData = [] {
      static const BufferAllocationInfo kBuffers[] = {
          BufferAllocationInfo::Result(16, 0),
          BufferAllocationInfo::Result(16, 1),
      };
      static constexpr int32_t kResults[] = {0, 1};
      auto* data = new StaticData;
      set_static_data_buffer_infos(data, kBuffers);
      set_static_data_num_buffers(data, 2);
      set_static_data_result_index_table(data, kResults);
      set_static_data_num_results(data, 2);
      set_static_data_thunk_run_impl(
          data, [](void** buffer_table, xla::ExecutableRunOptions*,
                   xla::cpu::RngState& rng_state) {
            rng_state.GetAndUpdateState(static_cast<uint64_t*>(buffer_table[0]),
                                        /*delta=*/5);
            rng_state.GetAndUpdateState(static_cast<uint64_t*>(buffer_table[1]),
                                        /*delta=*/11);
            return true;
          });
      return data;
    }();
    return *kData;
  }
};

class LegacyMultiThunkRngFunction final : public XlaCompiledCpuFunction {
 public:
  LegacyMultiThunkRngFunction() : XlaCompiledCpuFunction(GetStaticData()) {}

  absl::int128 out0() const { return ReadInt128(result_data(0)); }
  absl::int128 out1() const { return ReadInt128(result_data(1)); }
  absl::int128 out2() const { return ReadInt128(result_data(2)); }

 private:
  static const StaticData& GetStaticData() {
    static const StaticData* const kData = [] {
      static const BufferAllocationInfo kBuffers[] = {
          BufferAllocationInfo::Result(16, 0),
          BufferAllocationInfo::Result(16, 1),
          BufferAllocationInfo::Result(16, 2),
      };
      static constexpr int32_t kResults[] = {0, 1, 2};
      auto* data = new StaticData;
      set_static_data_buffer_infos(data, kBuffers);
      set_static_data_num_buffers(data, 3);
      set_static_data_result_index_table(data, kResults);
      set_static_data_num_results(data, 3);
      set_static_data_rng_state_deltas(data, {1, 3, 7});
      set_static_data_thunk_run_impl(
          data,
          [](void** buffer_table, xla::ExecutableRunOptions*,
             std::vector<std::unique_ptr<xla::cpu::RngState>>& rng_states) {
            if (rng_states.size() != 3) {
              return false;
            }
            rng_states[0]->GetAndUpdateState(
                static_cast<uint64_t*>(buffer_table[0]));
            rng_states[1]->GetAndUpdateState(
                static_cast<uint64_t*>(buffer_table[1]));
            rng_states[2]->GetAndUpdateState(
                static_cast<uint64_t*>(buffer_table[2]));
            return true;
          });
      return data;
    }();
    return *kData;
  }
};

class LegacyNoRngFunction final : public XlaCompiledCpuFunction {
 public:
  LegacyNoRngFunction() : XlaCompiledCpuFunction(GetStaticData()) {}

  uint64_t out() const {
    uint64_t value = 0;
    std::memcpy(&value, result_data(0), sizeof(value));
    return value;
  }

 private:
  static const StaticData& GetStaticData() {
    static const StaticData* const kData = [] {
      static const BufferAllocationInfo kBuffers[] = {
          BufferAllocationInfo::Result(8, 0),
      };
      static constexpr int32_t kResults[] = {0};
      auto* data = new StaticData;
      set_static_data_buffer_infos(data, kBuffers);
      set_static_data_num_buffers(data, 1);
      set_static_data_result_index_table(data, kResults);
      set_static_data_num_results(data, 1);
      set_static_data_rng_state_deltas(data, {});
      set_static_data_thunk_run_impl(
          data,
          [](void** buffer_table, xla::ExecutableRunOptions*,
             std::vector<std::unique_ptr<xla::cpu::RngState>>& rng_states) {
            if (!rng_states.empty()) {
              return false;
            }
            *static_cast<uint64_t*>(buffer_table[0]) = 42;
            return true;
          });
      return data;
    }();
    return *kData;
  }
};

TEST(XlaCompiledCpuFunctionTest, SharedRngStatePersistsAcrossRunsAndMoves) {
  SharedRngFunction fn;
  ASSERT_TRUE(fn.Run());
  const absl::int128 initial = fn.out0();
  EXPECT_EQ(fn.out1(), initial + 5);

  ASSERT_TRUE(fn.Run());
  EXPECT_EQ(fn.out0(), initial + 16);
  EXPECT_EQ(fn.out1(), initial + 21);

  SharedRngFunction moved(std::move(fn));
  ASSERT_TRUE(moved.Run());
  EXPECT_EQ(moved.out0(), initial + 32);
  EXPECT_EQ(moved.out1(), initial + 37);

  // A separate instance built from the same StaticData starts its own counter.
  SharedRngFunction independent;
  ASSERT_TRUE(independent.Run());
  EXPECT_EQ(independent.out0(), initial);
}

TEST(XlaCompiledCpuFunctionTest,
     LegacyMultiThunkForwardingSharesCounterAcrossRunsAndMoves) {
  LegacyMultiThunkRngFunction fn;
  ASSERT_TRUE(fn.Run());
  const absl::int128 initial = fn.out0();
  EXPECT_EQ(fn.out1(), initial + 1);
  EXPECT_EQ(fn.out2(), initial + 4);

  ASSERT_TRUE(fn.Run());
  EXPECT_EQ(fn.out0(), initial + 11);
  EXPECT_EQ(fn.out1(), initial + 12);
  EXPECT_EQ(fn.out2(), initial + 15);

  LegacyMultiThunkRngFunction moved(std::move(fn));
  ASSERT_TRUE(moved.Run());
  EXPECT_EQ(moved.out0(), initial + 22);
  EXPECT_EQ(moved.out1(), initial + 23);
  EXPECT_EQ(moved.out2(), initial + 26);

  LegacyMultiThunkRngFunction independent;
  ASSERT_TRUE(independent.Run());
  EXPECT_EQ(independent.out0(), initial);
}

TEST(XlaCompiledCpuFunctionTest, LegacySetterSupportsEmptyDeltas) {
  LegacyNoRngFunction fn;
  ASSERT_TRUE(fn.Run());
  EXPECT_EQ(fn.out(), 42);
}

}  // namespace
}  // namespace tensorflow
