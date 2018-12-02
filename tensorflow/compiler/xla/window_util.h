/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_WINDOW_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_WINDOW_UTIL_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace window_util {

// Creates a window with the given sizes in the dimensions and all strides set
// to 1.
Window MakeWindow(absl::Span<const int64> sizes);

// Creates a padding config with symmetrical padding in each dimension, of value
// given by sizes; e.g. {0, 1, 2} would create a R3 padding config that had zero
// pixels of padding in dimension 0, one pixel of padding symmetrically, on each
// side of dimension 1, and two pixels of padding symmetrically on dimension 2.
PaddingConfig MakeSymmetricPadding(absl::Span<const int64> sizes);

string ToString(const WindowDimension& dim);
string ToString(const Window& window);

// The below functions return true if the given field is set to have a
// non-trivial effect, e.g. having a stride means that the stride of some
// dimension is not one. Whether the proto field is populated is not a
// consideration.

bool HasStride(const Window& window);
bool HasPadding(const Window& window);
bool HasSymmetricPadding(const Window& window);
bool HasNegativePadding(const Window& window);

// As with HasSymmetricPadding(Window) above, returns whether the "padding low"
// is equivalent to the "padding high" for all dimensions, but works on a
// padding configuration.
bool HasSymmetricPadding(const PaddingConfig& padding_config);

bool HasBaseDilation(const Window& window);
bool HasWindowDilation(const Window& window);
bool HasDilation(const Window& window);

bool HasWindowReversal(const Window& window);
bool AllOrNoneReversed(const Window& window);

// Returns true if the given logical dimension is inactive in the sense that it
// has window bound 1, no striding and no padding.
bool IsInactiveWindowDimension(const Window& window, int64 logical_dim);

// Returns the new bound after dilation.
//
// If a window with the given bound in some dimension is dilated with the given
// dilation factor in that dimension, then the value returned is the bound for
// the array in that dimension after dilation.
//
// For a 1D array with 3 entries 1, 2, 3, a dilation factor of 2 yields a new
// window with values 1, x, 2, x, 3, where x indicates holes left by the
// dilation. So DilatedBound(3, 2) == 5.
int64 DilatedBound(int64 bound, int64 dilation);

// Returns the number of valid positions of a window with the given size and
// stride within an array with the given bound. This is the bound of an output
// array with one element per valid position of the window.
//
// For example, for arguments of (bound=5, window_size=2, stride=2), the
// returned value is 2. There are valid positions at offset 0 and offset 2,
// while offset 4 is not valid since the window's last entry would be at 5,
// which is beyond the bound of 5.
int64 StridedBound(int64 bound, int64 window_size, int64 stride);

}  // namespace window_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_WINDOW_UTIL_H_
