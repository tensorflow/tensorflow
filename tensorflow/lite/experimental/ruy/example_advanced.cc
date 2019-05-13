/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include <iostream>

#include "tensorflow/lite/experimental/ruy/ruy_advanced.h"

// Simple allocator for allocating pre-packed matrices.
class SimpleAllocator {
 public:
  void* AllocateBytes(std::size_t num_bytes) {
    char* p = new char[num_bytes];
    buffers_.emplace_back(p);
    return static_cast<void*>(p);
  }

 private:
  std::vector<std::unique_ptr<char[]>> buffers_;
};

void ExamplePrepack(ruy::Context* context) {
  const float lhs_data[] = {1, 2, 3, 4};
  const float rhs_data[] = {1, 2, 3, 4};
  float dst_data[4];

  // Set up the matrix layouts and spec.
  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kRowMajor, &lhs.layout);
  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &rhs.layout);
  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(2, 2, ruy::Order::kColMajor, &dst.layout);
  ruy::BasicSpec<float, float> spec;

  SimpleAllocator allocator;
  auto alloc_fn = [&allocator](std::size_t num_bytes) -> void* {
    return allocator.AllocateBytes(num_bytes);
  };

  // In this example, we pre-pack only the RHS, but either will work.
  // Note that we only need to set the data pointer for the matrix we are
  // pre-packing.
  ruy::PrepackedMatrix prepacked_rhs;
  rhs.data = rhs_data;
  ruy::PrePackForMul<ruy::kAllPaths>(lhs, rhs, spec, context, &dst,
                                     /*prepacked_lhs=*/nullptr, &prepacked_rhs,
                                     alloc_fn);

  // No data will be read from the RHS input matrix when using a pre-packed RHS.
  rhs.data = nullptr;
  lhs.data = lhs_data;
  dst.data = dst_data;
  ruy::MulWithPrepacked<ruy::kAllPaths>(lhs, rhs, spec, context, &dst,
                                        /*prepacked_lhs=*/nullptr,
                                        &prepacked_rhs);
  rhs.data = rhs_data;

  // Print out the results.
  std::cout << "Example Mul with pre-packing RHS, float:\n";
  std::cout << "LHS:\n" << lhs;
  std::cout << "RHS:\n" << rhs;
  std::cout << "Result:\n" << dst << "\n";
}

int main() {
  ruy::Context context;
  ExamplePrepack(&context);
}
