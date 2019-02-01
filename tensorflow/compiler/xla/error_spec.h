/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_ERROR_SPEC_H_
#define TENSORFLOW_COMPILER_XLA_ERROR_SPEC_H_

namespace xla {

// Structure describing permissible absolute and relative error bounds.
struct ErrorSpec {
  explicit ErrorSpec(float aabs, float arel = 0, bool relaxed_nans = false)
      : abs(aabs), rel(arel), relaxed_nans(relaxed_nans) {}

  float abs;  // Absolute error bound.
  float rel;  // Relative error bound.

  // If relaxed_nans is true then any result is valid if we are expecting NaNs.
  // In effect, this allows the tested operation to produce incorrect results
  // for inputs outside its mathematical domain.
  bool relaxed_nans;

  // If this is true, then we treat each +/-inf in the actual result as
  // equivalent to our choice of either +/-inf or the min/max floating-point
  // value.
  //
  // If the expected result is +/-inf, the actual result must still be +/-inf.
  //
  // In effect, this allows the tested operation to overflow, so long as it's
  // overflowing on "large" values.
  //
  // (We could have a symmetric more_infs_ok flag if necessary; right now it
  // appears not to be.)
  bool fewer_infs_ok = false;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ERROR_SPEC_H_
