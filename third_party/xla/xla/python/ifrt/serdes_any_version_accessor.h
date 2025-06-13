/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_SERDES_ANY_VERSION_ACCESSOR_H_
#define XLA_PYTHON_IFRT_SERDES_ANY_VERSION_ACCESSOR_H_

#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {

// Accessor for `SerDesVersion` that allows getting any SerDes version. Used for
// the layers that require full control over SerDes version selection.
class SerDesAnyVersionAccessor {
 public:
  static SerDesVersion GetMinimum() { return SerDesVersion::minimum(); }

  static SerDesVersion Get(SerDesVersionNumber version_number) {
    return SerDesVersion(version_number);
  }
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SERDES_ANY_VERSION_ACCESSOR_H_
