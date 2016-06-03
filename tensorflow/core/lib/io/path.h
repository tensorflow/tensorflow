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

#ifndef TENSORFLOW_LIB_IO_PATH_H_
#define TENSORFLOW_LIB_IO_PATH_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
class StringPiece;
namespace io {

// Utility routines for processing filenames

// Join multiple paths together, without introducing unnecessary path
// separators.
// For example:
//
//  Arguments                  | JoinPath
//  ---------------------------+----------
//  '/foo', 'bar'              | /foo/bar
//  '/foo/', 'bar'             | /foo/bar
//  '/foo', '/bar'             | /foo/bar
//
// Usage:
// string path = io::JoinPath("/mydir", filename);
// string path = io::JoinPath(FLAGS_test_srcdir, filename);
string JoinPath(StringPiece part1, StringPiece part2);

// Return true if path is absolute.
bool IsAbsolutePath(StringPiece path);

// Returns the part of the path before the final "/".  If there is a single
// leading "/" in the path, the result will be the leading "/".  If there is
// no "/" in the path, the result is the empty prefix of the input.
StringPiece Dirname(StringPiece path);

// Returns the part of the path after the final "/".  If there is no
// "/" in the path, the result is the same as the input.
StringPiece Basename(StringPiece path);

// Returns the part of the basename of path after the final ".".  If
// there is no "." in the basename, the result is empty.
StringPiece Extension(StringPiece path);

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_PATH_H_
