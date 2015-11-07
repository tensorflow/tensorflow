#ifndef TENSORFLOW_STREAM_EXECUTOR_PLATFORM_PORT_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLATFORM_PORT_H_

#include "tensorflow/core/platform/port.h"

namespace perftools {
namespace gputools {

using tensorflow::int8;
using tensorflow::int16;
using tensorflow::int32;
using tensorflow::int64;

using tensorflow::uint8;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::uint64;

#if !defined(PLATFORM_GOOGLE)
using std::string;
#endif

#if !defined(COMPILER_MSVC)
#define ARRAYSIZE(a)              \
    ((sizeof(a) / sizeof(*(a))) / \
    static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))
#endif

using tensorflow::LinkerInitialized;
using tensorflow::LINKER_INITIALIZED;

#define SE_FALLTHROUGH_INTENDED TF_FALLTHROUGH_INTENDED

}  // namespace gputools
}  // namespace perftools

#define SE_DISALLOW_COPY_AND_ASSIGN TF_DISALLOW_COPY_AND_ASSIGN
#define SE_MUST_USE_RESULT TF_MUST_USE_RESULT

#endif  // TENSORFLOW_STREAM_EXECUTOR_PLATFORM_PORT_H_
