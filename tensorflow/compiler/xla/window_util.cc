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

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace window_util {

Window MakeWindow(absl::Span<const int64> sizes) {
  Window window;
  for (int64 size : sizes) {
    auto* dimension = window.add_dimensions();
    dimension->set_size(size);
    dimension->set_stride(1);
    dimension->set_base_dilation(1);
    dimension->set_window_dilation(1);
  }
  return window;
}

Window MakeWindow(absl::Span<const int64> sizes,
                  absl::Span<const int64> strides) {
  Window window;
  CHECK_EQ(sizes.size(), strides.size());
  for (auto nb = 0; nb < sizes.size(); ++nb) {
    auto* dimension = window.add_dimensions();
    dimension->set_size(sizes[nb]);
    dimension->set_stride(strides[nb]);
    dimension->set_base_dilation(1);
    dimension->set_window_dilation(1);
  }
  return window;
}

PaddingConfig MakeSymmetricPadding(absl::Span<const int64> sizes) {
  PaddingConfig config;
  for (int64 size : sizes) {
    auto* dimension = config.add_dimensions();
    dimension->set_edge_padding_low(size);
    dimension->set_edge_padding_high(size);
  }
  return config;
}

/* static */ string ToString(const WindowDimension& dim) {
  using absl::StrAppend;
  using absl::StrCat;
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
  using absl::StrAppend;
  using absl::StrCat;

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

  if (window.dimensions_size() > 0) {
    add_field("size",
              [](const WindowDimension& dim) { return StrCat(dim.size()); });
  }
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

bool HasSymmetricPadding(const Window& window) {
  return absl::c_all_of(window.dimensions(), [](const WindowDimension& dim) {
    return dim.padding_low() == dim.padding_high();
  });
}

bool HasSymmetricPadding(const PaddingConfig& padding_config) {
  return absl::c_all_of(padding_config.dimensions(),
                        [](const PaddingConfig::PaddingConfigDimension& dim) {
                          return dim.edge_padding_low() ==
                                 dim.edge_padding_high();
                        });
}

bool HasNegativePadding(const Window& window) {
  return absl::c_any_of(window.dimensions(), [](const WindowDimension& dim) {
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

bool AllOrNoneReversed(const Window& window) {
  if (window.dimensions().empty()) {
    return true;
  }
  bool reversed = window.dimensions()[0].window_reversal();
  return absl::c_all_of(window.dimensions(), [&](const WindowDimension& dim) {
    return dim.window_reversal() == reversed;
  });
}

bool HasDilation(const Window& window) {
  return HasBaseDilation(window) || HasWindowDilation(window);
}

bool IsTrivialWindowDimension(const WindowDimension& window_dimension) {
  return window_dimension.size() == 1 && window_dimension.stride() == 1 &&
         window_dimension.padding_low() == 0 &&
         window_dimension.padding_high() == 0 &&
         window_dimension.window_dilation() == 1 &&
         window_dimension.base_dilation() == 1;
}

bool HasOverlappingWindow(const Window& window) {
  for (const auto& dim : window.dimensions()) {
    if (dim.size() > dim.stride()) {
      return true;
    }
  }
  return false;
}

int64 DilatedBound(int64 bound, int64 dilation) {
  CHECK_GE(bound, 0);
  CHECK_GE(dilation, 1);
  if (bound == 0) {
    return 0;
  }

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

  if (bound == 0 || window_size > bound) {
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
