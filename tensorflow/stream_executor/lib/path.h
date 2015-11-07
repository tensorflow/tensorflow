#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_PATH_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_PATH_H_

#include "tensorflow/stream_executor/lib/stringpiece.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {
namespace port {

namespace internal {
// TODO(rspringer): Move to cc/implementation file.
// Not part of the public API.
string JoinPathImpl(std::initializer_list<port::StringPiece> paths);
}  // namespace internal

// Join multiple paths together.
// JoinPath unconditionally joins all paths together. For example:
//
//  Arguments                  | JoinPath
//  ---------------------------+---------------------
//  '/foo', 'bar'              | /foo/bar
//  '/foo/', 'bar'             | /foo/bar
//  '/foo', '/bar'             | /foo/bar
//  '/foo', '/bar', '/baz'     | /foo/bar/baz
//
// All paths will be treated as relative paths, regardless of whether or not
// they start with a leading '/'.  That is, all paths will be concatenated
// together, with the appropriate path separator inserted in between.
// Arguments must be convertible to port::StringPiece.
//
// Usage:
// string path = file::JoinPath("/var/log", dirname, filename);
// string path = file::JoinPath(FLAGS_test_srcdir, filename);
template <typename... T>
inline string JoinPath(const T&... args) {
  return internal::JoinPathImpl({args...});
}

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_PATH_H_
