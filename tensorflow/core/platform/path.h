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

#ifndef TENSORFLOW_CORE_PLATFORM_PATH_H_
#define TENSORFLOW_CORE_PLATFORM_PATH_H_

#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {
namespace internal {
std::string JoinPathImpl(std::initializer_list<tensorflow::StringPiece> paths);
}

// Utility routines for processing filenames

#ifndef SWIG  // variadic templates
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
// string path = io::JoinPath("/full", "path", "to", "filename");
template <typename... T>
std::string JoinPath(const T&... args) {
  return internal::JoinPathImpl({args...});
}
#endif /* SWIG */

// Return true if path is absolute.
bool IsAbsolutePath(tensorflow::StringPiece path);

// Returns the part of the path before the final "/".  If there is a single
// leading "/" in the path, the result will be the leading "/".  If there is
// no "/" in the path, the result is the empty prefix of the input.
tensorflow::StringPiece Dirname(tensorflow::StringPiece path);

// Returns the part of the path after the final "/".  If there is no
// "/" in the path, the result is the same as the input.
tensorflow::StringPiece Basename(tensorflow::StringPiece path);

// Returns the part of the basename of path after the final ".".  If
// there is no "." in the basename, the result is empty.
tensorflow::StringPiece Extension(tensorflow::StringPiece path);

// Returns the largest common subpath of `paths`.
//
// For example, for "/alpha/beta/gamma" and "/alpha/beta/ga" returns
// "/alpha/beta/". For "/alpha/beta/gamma" and "/alpha/beta/gamma" returns
// "/alpha/beta/".
//
// Does not perform any path normalization.
std::string CommonPathPrefix(absl::Span<std::string const> paths);

// Collapse duplicate "/"s, resolve ".." and "." path elements, remove
// trailing "/".
//
// NOTE: This respects relative vs. absolute paths, but does not
// invoke any system calls (getcwd(2)) in order to resolve relative
// paths with respect to the actual working directory.  That is, this is purely
// string manipulation, completely independent of process state.
std::string CleanPath(tensorflow::StringPiece path);

// Populates the scheme, host, and path from a URI. scheme, host, and path are
// guaranteed by this function to point into the contents of uri, even if
// empty.
//
// Corner cases:
// - If the URI is invalid, scheme and host are set to empty strings and the
//   passed string is assumed to be a path
// - If the URI omits the path (e.g. file://host), then the path is left empty.
void ParseURI(tensorflow::StringPiece uri, tensorflow::StringPiece* scheme,
              tensorflow::StringPiece* host, tensorflow::StringPiece* path);

// Creates a URI from a scheme, host, and path. If the scheme is empty, we just
// return the path.
std::string CreateURI(tensorflow::StringPiece scheme,
                      tensorflow::StringPiece host,
                      tensorflow::StringPiece path);

// Creates a temporary file name with an extension.
std::string GetTempFilename(const std::string& extension);

// Reads the TEST_UNDECLARED_OUTPUTS_DIR environment variable, and if set
// assigns `dir` to the value. `dir` is not modified if the environment variable
// is unset. Returns true if the environment variable is set, otherwise false.
// Passing `dir` as nullptr, will just probe for the environment variable.
//
// Note: This function obviates the need to deal with Bazel's odd path decisions
// on Windows, and should be preferred over a simple `getenv`.
bool GetTestUndeclaredOutputsDir(std::string* dir);

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PATH_H_
