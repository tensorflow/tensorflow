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

// Legacy flags for XLA's layout_util module.

#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <vector>

#include "tensorflow/compiler/xla/legacy_flags/layout_util_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Pointers to the string value of the xla_default_layout flag and the flag
// descriptor, initialized via raw_flags_init.
static string* raw_flag;
static std::vector<tensorflow::Flag>* flag_list;
static std::once_flag raw_flags_init;

// Allocate *raw_flag.  Called via call_once(&raw_flags_init,...).
static void AllocateRawFlag() {
  raw_flag = new string;
  flag_list = new std::vector<tensorflow::Flag>({
      tensorflow::Flag(
          "xla_default_layout", raw_flag,
          "Default layout for Shapes in XLA. Valid values are: "
          "'minor2major', 'major2minor', 'random', 'random:<seed>'. "
          "For debugging purposes. If no seed (or 0) is given, a seed from "
          "random_device is used."),
  });
  ParseFlagsFromEnv(*flag_list);
}

// Parse text into *layout.
static bool ParseDefaultLayout(const string& text, DefaultLayout* layout) {
  bool result = true;
  std::vector<string> field = tensorflow::str_util::Split(text, ':');
  if (field.size() > 0) {
    if (field[0] == "random") {
      layout->dimension_order = DefaultLayout::DimensionOrder::kRandom;
      if (field.size() > 1) {
        uint64 seed = 0;
        result = tensorflow::strings::safe_strtou64(field[1], &seed);
        layout->seed = seed;
      }
    } else if (field[0] == "minor2major") {
      layout->dimension_order = DefaultLayout::DimensionOrder::kMinorToMajor;
    } else if (field[0] == "major2minor") {
      layout->dimension_order = DefaultLayout::DimensionOrder::kMajorToMinor;
    } else {
      result = false;
    }
  }
  return result;
}

// Pointer to the parsed value of the flags, initialized via flags_init.
static LayoutUtilFlags* flags;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  std::call_once(raw_flags_init, &AllocateRawFlag);
  flags = new LayoutUtilFlags;
  flags->xla_default_layout.dimension_order =
      DefaultLayout::DimensionOrder::kMajorToMinor;
  flags->xla_default_layout.seed = 0;
  if (!ParseDefaultLayout(*raw_flag, &flags->xla_default_layout)) {
    flags = nullptr;
  }
}

// Append to *append_to the flag definitions associated with XLA's layout_util
// module.
void AppendLayoutUtilFlags(std::vector<tensorflow::Flag>* append_to) {
  std::call_once(raw_flags_init, &AllocateRawFlag);
  append_to->insert(append_to->end(), flag_list->begin(), flag_list->end());
}

// Return a pointer to the LayoutUtilFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
LayoutUtilFlags* GetLayoutUtilFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace xla
