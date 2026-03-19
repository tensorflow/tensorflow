/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_GRID_H_
#define TENSORFLOW_CORE_TFRT_IFRT_GRID_H_

#include <ostream>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"

namespace tensorflow {
namespace ifrt_serving {

// Coordinates that identify a particular point in a 4-d grid (usually a TPU
// topology).
struct GridCoords {
  int dim[4];

  constexpr GridCoords(int d0, int d1, int d2, int d3) : dim{d0, d1, d2, d3} {}
  GridCoords() : GridCoords(0, 0, 0, 0) {}

  static GridCoords Zeroes() { return GridCoords(0, 0, 0, 0); }
  static GridCoords Ones() { return GridCoords(1, 1, 1, 1); }

  int operator[](int i) const {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, 4);
    return dim[i];
  }

  int& operator[](int i) {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, 4);
    return dim[i];
  }

  int Product() const { return dim[0] * dim[1] * dim[2] * dim[3]; }

  std::string ToString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const GridCoords& value) {
    absl::Format(&sink, "%s", value.ToString());
  }

  friend bool operator==(const GridCoords& a, const GridCoords& b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
  }

  friend std::ostream& operator<<(std::ostream& os, const GridCoords& c) {
    return os << c.ToString();
  }

  template <typename H>
  friend H AbslHashValue(H h, const GridCoords& c) {
    return H::combine(std::move(h), c[0], c[1], c[2], c[3]);
  }
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_GRID_H_
