#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_H_

#include "tensorflow/core/public/status.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace perftools {
namespace gputools {
namespace port {

using tensorflow::Status;

#define SE_CHECK_OK(val) \
  CHECK_EQ(::perftools::gputools::port::Status::OK(), (val))
#define SE_ASSERT_OK(val) \
  ASSERT_EQ(::perftools::gputools::port::Status::OK(), (val))

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STATUS_H_
