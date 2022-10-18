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

// An iterator yields a sequence of key/value pairs from a source.
// The following class defines the interface.  Multiple implementations
// are provided by this library.  In particular, iterators are provided
// to access the contents of a Table or a DB.
//
// Multiple threads can invoke const methods on an Iterator without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same Iterator must use
// external synchronization.

#ifndef TENSORFLOW_CORE_LIB_IO_ITERATOR_H_
#define TENSORFLOW_CORE_LIB_IO_ITERATOR_H_

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/tsl/lib/io/iterator.h"

namespace tensorflow {
namespace table {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::table::Iterator;
using tsl::table::NewEmptyIterator;
using tsl::table::NewErrorIterator;
// NOLINTEND(misc-unused-using-decls)
}  // namespace table
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_ITERATOR_H_
