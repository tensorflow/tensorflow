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

#ifndef TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_LAYOUT_UTIL_FLAGS_H_
#define TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_LAYOUT_UTIL_FLAGS_H_

// Legacy flags for the XLA's layout_util module.

#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// The default layout for all newly created shapes. Specified by the flag
// --xla_default_layout.
struct DefaultLayout {
  enum class DimensionOrder {
    kRandom,
    kMinorToMajor,
    kMajorToMinor,
  };

  DimensionOrder dimension_order;
  size_t seed;
};

// Append to *flag_list the flag definitions associated with XLA's layout_util
// module.
void AppendLayoutUtilFlags(std::vector<tensorflow::Flag>* flag_list);

// The values of flags associated with XLA's layout_util module.
typedef struct {
  // Default layout for Shapes in XLA.  Valid values are:  'minor2major',
  // 'major2minor', 'random', 'random:<seed>'.  For debugging purposes.  If no
  // seed (or 0) is given, a seed from random_device is used.
  DefaultLayout xla_default_layout;
} LayoutUtilFlags;

// Return a pointer to the LayoutFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
LayoutUtilFlags* GetLayoutUtilFlags();

}  // namespace legacy_flags
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_LAYOUT_UTIL_FLAGS_H_
