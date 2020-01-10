#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_INITIALIZE_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_INITIALIZE_H_

#include "tensorflow/stream_executor/platform/port.h"

#if defined(PLATFORM_GOOGLE)
#else

#undef REGISTER_MODULE_INITIALIZER

namespace perftools {
namespace gputools {
namespace port {

class Initializer {
 public:
  typedef void (*InitializerFunc)();
  explicit Initializer(InitializerFunc func) { func(); }
};

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#define REGISTER_INITIALIZER(type, name, body)                               \
  static void google_init_##type##_##name() { body; }                        \
  perftools::gputools::port::Initializer google_initializer_##type##_##name( \
      google_init_##type##_##name)

#define REGISTER_MODULE_INITIALIZER(name, body) \
  REGISTER_INITIALIZER(module, name, body)

#endif  // !defined(PLATFORM_GOOGLE)

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_INITIALIZE_H_
