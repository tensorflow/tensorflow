#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_PROCESS_STATE_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_PROCESS_STATE_H_

#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {
namespace port {

string Hostname();
bool GetCurrentDirectory(string* dir);

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_PROCESS_STATE_H_
