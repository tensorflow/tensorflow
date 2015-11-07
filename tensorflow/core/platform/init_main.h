#ifndef TENSORFLOW_PLATFORM_INIT_MAIN_H_
#define TENSORFLOW_PLATFORM_INIT_MAIN_H_

namespace tensorflow {
namespace port {

// Platform-specific initialization routine that may be invoked by a
// main() program that uses TensorFlow.
//
// Default implementation does nothing.
void InitMain(const char* usage, int* argc, char*** argv);

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_INIT_MAIN_H_
