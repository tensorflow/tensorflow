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

#include "tensorflow/compiler/xla/window_util.h"

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace xla {
namespace window_util {

/* static */ string ToString(const WindowDimension& dim) {
  using tensorflow::strings::StrAppend;
  using tensorflow::strings::StrCat;
  string str = StrCat("(size=", dim.size());
  if (dim.stride() != 1) {
    StrAppend(&str, ",stride=", dim.stride());
  }
  if (dim.padding_low() != 0) {
    StrAppend(&str, ",padding_low=", dim.padding_low());
  }
  if (dim.padding_high() != 0) {
    StrAppend(&str, ",padding_high=", dim.padding_high());
  }
  if (dim.base_dilation() != 1) {
    StrAppend(&str, ",base_dilation=", dim.base_dilation());
  }
  if (dim.window_dilation() != 1) {
    StrAppend(&str, ",window_dilation=", dim.window_dilation());
  }
  if (dim.window_reversal()) {
    StrAppend(&str, ",window_reversal");
  }
  StrAppend(&str, ")");
  return str;
}

string ToString(const Window& window) {
  using tensorflow::strings::StrAppend;
  using tensorflow::strings::StrCat;

  string str;
  const auto add_field =
      [&](const char* heading,
          std::function<string(const WindowDimension&)> format) {
        StrAppend(&str, heading, "=");
        const char* prefix = "";
        for (const auto& window_dimension : window.dimensions()) {
          StrAppend(&str, prefix, format(window_dimension));
          prefix = "x";
        }
      };

  add_field("size",
            [](const WindowDimension& dim) { return StrCat(dim.size()); });
  if (HasStride(window)) {
    add_field(" stride",
              [](const WindowDimension& dim) { return StrCat(dim.stride()); });
  }
  if (HasPadding(window)) {
    add_field(" pad", [](const WindowDimension& dim) {
      return StrCat(dim.padding_low(), "_", dim.padding_high());
    });
  }
  if (HasBaseDilation(window)) {
    add_field(" lhs_dilate", [](const WindowDimension& dim) {
      return StrCat(dim.base_dilation());
    });
  }
  if (HasWindowDilation(window)) {
    add_field(" rhs_dilate", [](const WindowDimension& dim) {
      return StrCat(dim.window_dilation());
    });
  }
  if (HasWindowReversal(window)) {
    add_field(" rhs_reversal", [](const WindowDimension& dim) {
      return StrCat(dim.window_reversal() ? 1 : 0);
    });
  }
  return str;
}

bool HasStride(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.stride() != 1) {
      return true;
    }
  }
  return false;
}

bool HasPadding(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.padding_low() != 0 || dim.padding_high() != 0) {
      return true;
    }
  }
  return false;
}

bool HasEvenPadding(const Window& window) {
  return std::all_of(window.dimensions().begin(), window.dimensions().end(),
                     [](const WindowDimension& dim) {
                       return dim.padding_low() == dim.padding_high();
                     });
}

bool HasNegativePadding(const Window& window) {
  return std::any_of(window.dimensions().begin(), window.dimensions().end(),
                     [](const WindowDimension& dim) {
                       return dim.padding_low() < 0 || dim.padding_high() < 0;
                     });
}

bool HasBaseDilation(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.base_dilation() != 1) {
      return true;
    }
  }
  return false;
}

bool HasWindowDilation(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.window_dilation() != 1) {
      return true;
    }
  }
  return false;
}

bool HasWindowReversal(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.window_reversal()) {
      return true;
    }
  }
  return false;
}

bool HasDilation(const Window& window) {
  return HasBaseDilation(window) || HasWindowDilation(window);
}

bool IsInactiveWindowDimension(const Window& window, int64 logical_dim) {
  const WindowDimension& window_dim = window.dimensions(logical_dim);
  return window_dim.size() == 1 && window_dim.stride() == 1 &&
         window_dim.padding_low() == 0 && window_dim.padding_high() == 0;
}

int64 DilatedBound(int64 bound, int64 dilation) {
  CHECK_GE(bound, 0);
  CHECK_GE(dilation, 1);

  // Suppose the array has three entries 123 and the dilation factor is 4. Then
  // the dilated array has 9 entries 1xxx2xxx3. Here, each original entry except
  // the last expands into 4 entries, so that is (bound - 1) * dilation. Then we
  // add 1 to account for the final input element.
  return (bound - 1) * dilation + 1;
}

int64 StridedBound(int64 bound, int64 window_size, int64 stride) {
  CHECK_GE(window_size, 0);
  CHECK_GE(bound, 0);
  CHECK_GE(stride, 1);

  if (window_size > bound) {
    return 0;
  }

  // Without considering stride, the maximum valid offset is bound -
  // window_size. Taking stride into account, the valid offsets then have the
  // form q * stride for q = 0, ..., Q such that q * stride <= bound -
  // window_size. This implies that Q equals floor(bound - window_size /
  // stride). There are Q + 1 valid values of q, yielding the formula below.
  return (bound - window_size) / stride + 1;
}

}  // namespace window_util
}  // namespace xla
