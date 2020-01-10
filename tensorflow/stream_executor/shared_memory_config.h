// This file defines a uniform interface to configuration options for shared
// memory for supported devices. As with many StreamExecutor-supported features,
// support for the options defined herein is device-dependent.
#ifndef TENSORFLOW_STREAM_EXECUTOR_SHARED_MEMORY_CONFIG_H_
#define TENSORFLOW_STREAM_EXECUTOR_SHARED_MEMORY_CONFIG_H_

namespace perftools {
namespace gputools {

// SharedMemoryConfig enum describes potential widths of shared memory banks for
// a device or kernel.
enum class SharedMemoryConfig {
  kDefault,    // Use the device default configuration.
  kFourByte,   // Sets shared memory banks to be four bytes wide.
  kEightByte,  // Sets shared memory banks to be eight bytes wide.
};

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_SHARED_MEMORY_CONFIG_H_
