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

#include "tensorflow/compiler/xla/runtime/results.h"

#include <memory>
#include <optional>
#include <utility>

#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/runtime/types.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/test_benchmark.h"

namespace xla {
namespace runtime {

TEST(ResultsTest, ResultConverterSet) {
  std::optional<int32_t> value;

  auto ret_error = [&](const absl::Status& status) { ASSERT_TRUE(false); };

  auto ret_i32 = [&](unsigned, const Type* type, const Type*, void* ret) {
    auto* scalar = llvm::dyn_cast<ScalarType>(type);
    if (scalar && scalar->type() == PrimitiveType::S32) {
      value = *reinterpret_cast<int32_t*>(ret);
      return success();
    }
    return failure();
  };

  auto s32 = std::make_unique<ScalarType>(PrimitiveType::S32);
  auto s64 = std::make_unique<ScalarType>(PrimitiveType::S64);

  ResultConverterSet converter(ret_error, ret_i32);

  // S64 conversion is not supported.
  ASSERT_TRUE(failed(converter.ReturnValue(0, s64.get(), s64.get(), nullptr)));

  // Check that int32_t value was successfully returned.
  int32_t i32 = 42;
  ASSERT_TRUE(succeeded(converter.ReturnValue(0, s32.get(), s32.get(), &i32)));
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, i32);
}

TEST(ResultsTest, MoveOnlyResultConverterSet) {
  {  // Move-only capture in error converter.
    auto ptr = std::make_unique<int32_t>(42);
    auto ret_error = [ptr = std::move(ptr)](const absl::Status& status) {};
    auto ret_value = [](unsigned, const Type*, const Type*, void*) {
      return success();
    };

    ResultConverterSet converter(std::move(ret_error), ret_value);
  }

  {  // Move-only capture in value converter.
    auto ptr = std::make_unique<int32_t>(42);
    auto ret_error = [](const absl::Status& status) {};
    auto ret_value = [ptr = std::move(ptr)](unsigned, const Type*, const Type*,
                                            void* ret) { return success(); };

    ResultConverterSet converter(ret_error, std::move(ret_value));
  }
}

//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

static void ReturnError(const absl::Status& status) {
  assert(false && "Unexpected call to `ReturnError`");
}

static LogicalResult ReturnI32(unsigned, const Type* type, const Type*, void*) {
  auto* scalar = llvm::dyn_cast<ScalarType>(type);
  return (scalar && scalar->type() == PrimitiveType::S32) ? success()
                                                          : failure();
}

static void BM_RetI32(benchmark::State& state) {
  ResultConverterSet converter(ReturnError, ReturnI32);

  auto s32 = std::make_unique<ScalarType>(PrimitiveType::S32);

  for (auto _ : state) {
    auto converted = converter.ReturnValue(0, s32.get(), s32.get(), nullptr);
    benchmark::DoNotOptimize(converted);
  }
}

BENCHMARK(BM_RetI32);

}  // namespace runtime
}  // namespace xla
