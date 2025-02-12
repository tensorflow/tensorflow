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

#ifndef TENSORFLOW_CORE_LIB_IO_FORMAT_H_
#define TENSORFLOW_CORE_LIB_IO_FORMAT_H_

#include "xla/tsl/lib/io/format.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace table {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::table::BlockContents;
using tsl::table::BlockHandle;
using tsl::table::kBlockTrailerSize;
using tsl::table::kTableMagicNumber;
using tsl::table::ReadBlock;
// NOLINTEND(misc-unused-using-decls)
}  // namespace table
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_FORMAT_H_
