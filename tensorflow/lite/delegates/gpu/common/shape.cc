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
#include "tensorflow/lite/delegates/gpu/common/shape.h"

#include <stdint.h>

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace tflite {
namespace gpu {
namespace {

struct GetAxisByIndexFunc {
  template <Layout T>
  Axis operator()() const {
    return GetAxis<T>(index);
  }
  int32_t index;
};

struct GetIndexByAxisFunc {
  template <Layout T>
  int operator()() const {
    return GetAxisIndex<T>(axis);
  }
  Axis axis;
};

struct NumAxisFunc {
  template <Layout T>
  int operator()() const {
    return Size<T>();
  }
};

}  // namespace

std::string ToString(Axis axis) {
  switch (axis) {
    case Axis::BATCH:
      return "batch";
    case Axis::CHANNELS:
      return "channels";
    case Axis::INPUT_CHANNELS:
      return "input_channels";
    case Axis::OUTPUT_CHANNELS:
      return "output_channels";
    case Axis::HEIGHT:
      return "height";
    case Axis::WIDTH:
      return "width";
    case Axis::VALUE:
      return "value";
    case Axis::DEPTH:
      return "depth";
    case Axis::UNKNOWN:
      return "unknown";
  }
  return "undefined";
}

std::string ToString(Layout layout) {
  switch (layout) {
    case Layout::SCALAR:
      return "scalar";
    case Layout::LINEAR:
      return "linear";
    case Layout::HW:
      return "hw";
    case Layout::HWD:
      return "hwd";
    case Layout::CHW:
      return "chw";
    case Layout::HWC:
      return "hwc";
    case Layout::HWDC:
      return "hwdc";
    case Layout::OHWI:
      return "ohwi";
    case Layout::IHWO:
      return "ihwo";
    case Layout::OIHW:
      return "oihw";
    case Layout::IOHW:
      return "iohw";
    case Layout::BHWC:
      return "bhwc";
    case Layout::BHWDC:
      return "bhwdc";
    case Layout::OHWDI:
      return "ohwdi";
    case Layout::HWIO:
      return "hwio";
    case Layout::UNKNOWN:
      return "unknown";
  }
  return "undefined";
}

Axis GetAxis(Layout layout, int32_t index) {
  return DispatchByLayout(layout, GetAxisByIndexFunc{index});
}

int GetAxisIndex(Layout layout, Axis axis) {
  return DispatchByLayout(layout, GetIndexByAxisFunc{axis});
}

bool HasAxis(Layout layout, Axis axis) {
  return GetAxisIndex(layout, axis) >= 0;
}

int Size(Layout layout) { return DispatchByLayout(layout, NumAxisFunc()); }

std::string ToString(const Shape& s) {
  return absl::StrCat("{", ToString(s.layout), ", {",
                      absl::StrJoin(s.dimensions, ", "), "}}");
}

}  // namespace gpu
}  // namespace tflite
