/* Copyright 2017 The OpenXLA Authors.

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

// Tests that slice operations can be performed.

#include <memory>

#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_builder.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class SliceTest : public ClientLibraryTestBase {};

XLA_TEST_F(SliceTest, Slice2D) {
  XlaBuilder builder("slice_2d");
  auto original = ConstantR2<float>(
      &builder,
      {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}});
  Slice(original, {2, 1}, {4, 3}, {1, 1});

  Array2D<float> expected({{8.0f, 9.0f}, {11.0f, 12.0f}});
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

XLA_TEST_F(SliceTest, Slice3D) {
  XlaBuilder builder("slice_3d");
  Array3D<float> array_3d(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
  auto original = ConstantR3FromArray3D<float>(&builder, array_3d);
  Slice(original, {0, 0, 1}, {2, 1, 2}, {1, 1, 1});

  Array3D<float> expected_3d({{{2.0f}}, {{6.0f}}});
  ComputeAndCompareR3<float>(&builder, expected_3d, {}, ErrorSpec(0.000001));
}

}  // namespace
}  // namespace xla
