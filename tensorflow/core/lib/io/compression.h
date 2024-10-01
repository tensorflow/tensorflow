/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_IO_COMPRESSION_H_
#define TENSORFLOW_CORE_LIB_IO_COMPRESSION_H_

#include "xla/tsl/lib/io/compression.h"

namespace tensorflow {
namespace io {
namespace compression {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::io::compression::kGzip;
using tsl::io::compression::kNone;
using tsl::io::compression::kSnappy;
using tsl::io::compression::kZlib;
// NOLINTEND(misc-unused-using-decls)
}  // namespace compression
}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_COMPRESSION_H_
