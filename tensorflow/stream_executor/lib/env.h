#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_ENV_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_ENV_H_

#include "tensorflow/core/public/env.h"
#include "tensorflow/stream_executor/lib/stringpiece.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {
namespace port {

using tensorflow::Env;
using tensorflow::ReadFileToString;
using tensorflow::Thread;
using tensorflow::WriteStringToFile;

inline bool FileExists(const string& filename) {
  return Env::Default()->FileExists(filename);
}

inline bool FileExists(const port::StringPiece& filename) {
  return Env::Default()->FileExists(filename.ToString());
}

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_ENV_H_
