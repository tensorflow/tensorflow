/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/tensor_cord.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

void DoNothingReleaser(void*) {}

TEST(TensorCordTest, Empty) {
  TensorCord tc;
  EXPECT_EQ(tc.size(), 0);
  EXPECT_EQ(tc.chunk_begin(), tc.chunk_end());
  auto chunks = tc.Chunks();
  EXPECT_EQ(chunks.begin(), chunks.end());
}

TEST(TensorCordTest, ViewOfValue) {
  TensorCord tc("abc", &DoNothingReleaser, nullptr);
  EXPECT_EQ(*tc.chunk_begin(), "abc");
  auto it = tc.chunk_begin();
  EXPECT_EQ(*it, "abc");
  ++it;
  EXPECT_EQ(it, tc.chunk_end());
}

TEST(TensorCordTest, Chunks) {
  TensorCord tc("abc", &DoNothingReleaser, nullptr);
  int counter = 0;
  for (auto string_piece : tc.Chunks()) {
    EXPECT_EQ(string_piece, "abc");
    ++counter;
  }
  EXPECT_EQ(counter, 1);
}

// This function  takes an arg-less std::function that may have a closure, and
// creates a std::function with no closure: one that can be cast
// directly to a (*)(void*) function pointer that takes the original function.
// Use it this way:
//
//  void callback_with_arg((* fn)(void*), void* arg) { fn(arg); }
//
//  auto fn = [&]() { ... }
//  auto thunk = CreateThunkFor(fn);
//  callback_with_arg(thunk, &fn);
//
// Idea from:
//   http://bannalia.blogspot.com/2016/07/passing-capturing-c-lambda-functions-as.html
template <typename T>
CordRepReleaser CreateThunkFor(const T& fn) {
  return [](void* ptr) { (*static_cast<T*>(ptr))(); };
}

TEST(TensorCordTest, Copy) {
  int cleaned = 0;
  auto cleaner = [&cleaned]() { ++cleaned; };
  auto thunk = CreateThunkFor(cleaner);
  TensorCord tc_copy;
  string a = "abc";
  {
    TensorCord tc(a, thunk, &cleaner);
    tc_copy = tc;
  }
  auto it = tc_copy.chunk_begin();
  EXPECT_EQ(*it, "abc");
  ++it;
  EXPECT_EQ(it, tc_copy.chunk_end());
  EXPECT_EQ(cleaned, 0);
  tc_copy = TensorCord();
  EXPECT_EQ(cleaned, 1);
}

TEST(TensorCordTest, AppendCord) {
  int cleaned_0 = 0;
  int cleaned_1 = 0;
  auto cleaner_0 = [&cleaned_0]() { ++cleaned_0; };
  auto cleaner_1 = [&cleaned_1]() { ++cleaned_1; };
  auto thunk_0 = CreateThunkFor(cleaner_0);
  auto thunk_1 = CreateThunkFor(cleaner_1);
  TensorCord tc_0("abc", thunk_0, &cleaner_0);
  TensorCord tc_1("cba", thunk_1, &cleaner_1);
  tc_0.Append(tc_1);
  EXPECT_EQ(string(tc_0), "abccba");
  auto it = tc_0.chunk_begin();
  EXPECT_EQ(*it, "abc");
  ++it;
  EXPECT_EQ(*it, "cba");
  ++it;
  EXPECT_EQ(it, tc_0.chunk_end());
  tc_1 = TensorCord();
  EXPECT_EQ(cleaned_0, 0);
  EXPECT_EQ(cleaned_1, 0);
  tc_0 = TensorCord();
  EXPECT_EQ(cleaned_0, 1);
  EXPECT_EQ(cleaned_1, 1);
}

TEST(TensorCordTest, AppendView) {
  int cleaned_0 = 0;
  int cleaned_1 = 0;
  auto cleaner_0 = [&cleaned_0]() { ++cleaned_0; };
  auto cleaner_1 = [&cleaned_1]() { ++cleaned_1; };
  auto thunk_0 = CreateThunkFor(cleaner_0);
  auto thunk_1 = CreateThunkFor(cleaner_1);
  TensorCord tc_0("abc", thunk_0, &cleaner_0);
  tc_0.Append("cba", thunk_1, &cleaner_1);
  EXPECT_EQ(string(tc_0), "abccba");
  auto it = tc_0.chunk_begin();
  EXPECT_EQ(*it, "abc");
  ++it;
  EXPECT_EQ(*it, "cba");
  ++it;
  EXPECT_EQ(it, tc_0.chunk_end());
  EXPECT_EQ(cleaned_0, 0);
  EXPECT_EQ(cleaned_1, 0);
  tc_0 = TensorCord();
  EXPECT_EQ(cleaned_0, 1);
  EXPECT_EQ(cleaned_1, 1);
}

TEST(TensorCordTest, Move) {
  int cleaned = 0;
  auto cleaner = [&cleaned]() { ++cleaned; };
  auto thunk = CreateThunkFor(cleaner);
  TensorCord tc_copy;
  string a = "abc";
  {
    TensorCord tc(a, thunk, &cleaner);
    tc_copy = std::move(tc);
  }
  EXPECT_EQ(tc_copy.size(), 3);
  auto it = tc_copy.chunk_begin();
  EXPECT_EQ(*it, "abc");
  ++it;
  EXPECT_EQ(it, tc_copy.chunk_end());
  EXPECT_EQ(cleaned, 0);
  tc_copy = TensorCord();
  EXPECT_EQ(tc_copy.size(), 0);
  EXPECT_EQ(cleaned, 1);
}

TEST(TensorCordTest, CopyConstructor) {
  int cleaned = 0;
  auto cleaner = [&cleaned]() { ++cleaned; };
  auto thunk = CreateThunkFor(cleaner);
  string a = "abc";
  TensorCord tc(a, thunk, &cleaner);
  TensorCord tc_copy(tc);
  EXPECT_EQ(tc.size(), 3);
  EXPECT_EQ(tc_copy.size(), 3);
  auto it = tc_copy.chunk_begin();
  EXPECT_EQ(*it, "abc");
  ++it;
  EXPECT_EQ(it, tc_copy.chunk_end());
  EXPECT_EQ(cleaned, 0);
  tc = TensorCord();
  EXPECT_EQ(cleaned, 0);
  tc_copy = TensorCord();
  EXPECT_EQ(cleaned, 1);
}

TEST(TensorCordTest, MoveConstructor) {
  int cleaned = 0;
  auto cleaner = [&cleaned]() { ++cleaned; };
  auto thunk = CreateThunkFor(cleaner);
  string a = "abc";
  TensorCord tc(a, thunk, &cleaner);
  TensorCord tc_copy(std::move(tc));
  EXPECT_EQ(tc_copy.size(), 3);
  auto it = tc_copy.chunk_begin();
  EXPECT_EQ(*it, "abc");
  ++it;
  EXPECT_EQ(it, tc_copy.chunk_end());
  EXPECT_EQ(cleaned, 0);
  tc_copy = TensorCord();
  EXPECT_EQ(cleaned, 1);
}

#ifdef PLATFORM_GOOGLE

void TensorCopyFromTensorBenchmark(benchmark::State& state, int num_elem,
                                   int string_size) {
  Tensor strings(DT_STRING, {num_elem});
  auto t = strings.flat<tstring>();
  for (int i = 0; i < num_elem; ++i) {
    t(i).insert(0, string_size, 'a');
  }
  for (auto _ : state) {
    testing::DoNotOptimize(tensor::DeepCopy(strings));
  }
}

void TensorCordFromTensorBenchmark(benchmark::State& state, int num_elem,
                                   int string_size) {
  Tensor strings(DT_STRING, {num_elem});
  auto t = strings.flat<tstring>();
  for (int i = 0; i < num_elem; ++i) {
    t(i).insert(0, string_size, 'a');
  }
  for (auto _ : state) {
    Tensor copy(DT_VARIANT, {num_elem});
    auto t_copy = copy.flat<Variant>();
    for (int i = 0; i < num_elem; ++i) {
      t_copy(i) = TensorCord(t(i), &strings);
    }
  }
}

void CordReleaser(void* cord_ptr) { delete static_cast<absl::Cord*>(cord_ptr); }

void TensorCordFromAbslCordBenchmark(benchmark::State& state, int num_elem,
                                     int string_size) {
  std::vector<absl::Cord> cords(num_elem);
  for (int i = 0; i < num_elem; ++i) {
    string s(string_size, 'a');
    cords[i] = s;
  }

  for (auto _ : state) {
    Tensor copy(DT_VARIANT, {num_elem});
    auto t_copy = copy.flat<Variant>();
    for (int i = 0; i < num_elem; ++i) {
      auto my_cord = new absl::Cord(cords[i]);
      t_copy(i) = TensorCord(*my_cord->chunk_begin(), CordReleaser, my_cord);
    }
  }
}

#define CreateBM(NUM_ELEM, STRING_SIZE)                                        \
  void BM_TensorCopyFromTensor_NumElem_##NUM_ELEM##_StringSize_##STRING_SIZE(  \
      benchmark::State& state) {                                               \
    TensorCopyFromTensorBenchmark(state, NUM_ELEM, STRING_SIZE);               \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_TensorCopyFromTensor_NumElem_##NUM_ELEM##_StringSize_##STRING_SIZE);  \
  void BM_TensorCordFromTensor_NumElem_##NUM_ELEM##_StringSize_##STRING_SIZE(  \
      benchmark::State& state) {                                               \
    TensorCordFromTensorBenchmark(state, NUM_ELEM, STRING_SIZE);               \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_TensorCordFromTensor_NumElem_##NUM_ELEM##_StringSize_##STRING_SIZE);  \
  void                                                                         \
      BM_TensorCordFromAbslCord_NumElem_##NUM_ELEM##_StringSize_##STRING_SIZE( \
          benchmark::State& state) {                                           \
    TensorCordFromAbslCordBenchmark(state, NUM_ELEM, STRING_SIZE);             \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_TensorCordFromAbslCord_NumElem_##NUM_ELEM##_StringSize_##STRING_SIZE);

#define CreateStringBMs(NUM_ELEM)           \
  CreateBM(NUM_ELEM, /*STRING_SIZE=*/16);   \
  CreateBM(NUM_ELEM, /*STRING_SIZE=*/32);   \
  CreateBM(NUM_ELEM, /*STRING_SIZE=*/128);  \
  CreateBM(NUM_ELEM, /*STRING_SIZE=*/1024); \
  CreateBM(NUM_ELEM, /*STRING_SIZE=*/4096);

CreateStringBMs(/*NUM_ELEM=*/1);
CreateStringBMs(/*NUM_ELEM=*/16);
CreateStringBMs(/*NUM_ELEM=*/32);
CreateStringBMs(/*NUM_ELEM=*/64);
CreateStringBMs(/*NUM_ELEM=*/128);

#endif  // PLATFORM_GOOGLE

}  // namespace

}  // namespace tensorflow
