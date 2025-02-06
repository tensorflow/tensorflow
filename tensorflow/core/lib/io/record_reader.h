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

#ifndef TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_
#define TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_

#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stringpiece.h"
#if !defined(IS_SLIM_BUILD)
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#endif  // IS_SLIM_BUILD
#include "xla/tsl/lib/io/record_reader.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::io::RecordReader;
using tsl::io::RecordReaderOptions;
using tsl::io::SequentialRecordReader;
// NOLINTEND(misc-unused-using-decls)
}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_
