/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace fuzzing {
namespace {

using ::fuzztest::Arbitrary;
using ::fuzztest::Domain;
using ::fuzztest::Filter;
using ::fuzztest::Map;
using ::fuzztest::VectorOf;

Domain<StatusOr<TensorShape>> AnyStatusOrTensorShape() {
  return Map(
      [](std::vector<int64_t> v) -> StatusOr<TensorShape> {
        TensorShape out;
        TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(v, &out));
        return out;
      },
      VectorOf(Arbitrary<int64_t>()));
}

}  // namespace

Domain<TensorShape> AnyValidTensorShape() {
  return Map([](StatusOr<TensorShape> t) { return *t; },
             Filter([](auto mfts) { return mfts.status().ok(); },
                    AnyStatusOrTensorShape()));
}

}  // namespace fuzzing
}  // namespace tensorflow
