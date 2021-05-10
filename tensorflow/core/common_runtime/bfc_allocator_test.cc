/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/bfc_allocator.h"

#include <algorithm>
#include <random>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// A fake SubAllocator to test the performance of BFCAllocator.
class FakeSubAllocator : public SubAllocator {
 public:
  FakeSubAllocator() : SubAllocator({}, {}), alloc_counter_(0) {}
  ~FakeSubAllocator() override {}

  // Alloc and Free functions are implemented as very cheap operations, so that
  // the benchmark can focus on the performance of BFCAllocator itself.
  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    *bytes_received = num_bytes;
    return reinterpret_cast<void*>(alloc_counter_++);
  }

  void Free(void* ptr, size_t num_bytes) override {}

  bool SupportsCoalescing() const override { return false; }

 private:
  int64 alloc_counter_;
};

void BM_Allocator(::testing::benchmark::State& state) {
  constexpr int kAllocSize = 1 << 14;
  const int kLongLivedObjects = state.range(0);
  const int kShortLivedObjects = state.range(1);

  FakeSubAllocator* sub_allocator = new FakeSubAllocator;
  BFCAllocator bfc_allocator(sub_allocator, 1 << 30, false, "GPU_0_bfc");

  string test_op_name = "test_op";
  ScopedMemoryDebugAnnotation annotation(test_op_name.data());

  // Allocate long lived objects.
  std::vector<void*> long_lived(kLongLivedObjects);
  for (int i = 0; i < kLongLivedObjects; i++) {
    long_lived[i] = bfc_allocator.AllocateRaw(1, kAllocSize);
  }
  std::vector<int> deallocation_order(kShortLivedObjects);
  for (int i = 0; i < kShortLivedObjects; i++) {
    deallocation_order[i] = i;
  }
  std::shuffle(deallocation_order.begin(), deallocation_order.end(),
               std::default_random_engine(0));

  // Allocate and deallocate short lived objects.
  std::vector<void*> short_lived(kShortLivedObjects);
  for (auto _ : state) {
    for (int i = 0; i < kShortLivedObjects; i++) {
      short_lived[i] = bfc_allocator.AllocateRaw(1, kAllocSize);
    }
    for (int i = 0; i < kShortLivedObjects; i++) {
      bfc_allocator.DeallocateRaw(short_lived[deallocation_order[i]]);
    }
  }
}
BENCHMARK(BM_Allocator)
    ->ArgPair(0, 256)
    ->ArgPair(1000, 256)
    ->ArgPair(10000, 256);

}  // namespace tensorflow
