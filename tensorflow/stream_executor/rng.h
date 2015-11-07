#ifndef TENSORFLOW_STREAM_EXECUTOR_RNG_H_
#define TENSORFLOW_STREAM_EXECUTOR_RNG_H_

#include <limits.h>
#include <complex>

#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

class Stream;
template <typename ElemT>
class DeviceMemory;

namespace rng {

// Random-number-generation support interface -- this can be derived from a GPU
// executor when the underlying platform has an RNG library implementation
// available. See StreamExecutor::AsRng().
// When a seed is not specified, the backing RNG will be initialized with the
// default seed for that implementation.
//
// Thread-hostile: see StreamExecutor class comment for details on
// thread-hostility.
class RngSupport {
 public:
  static const int kMinSeedBytes = 16;
  static const int kMaxSeedBytes = INT_MAX;

  // Releases any random-number-generation resources associated with this
  // support object in the underlying platform implementation.
  virtual ~RngSupport() {}

  // Populates a GPU memory allocation with random values appropriate for the
  // DeviceMemory element type; i.e. populates DeviceMemory<float> with random
  // float values.
  virtual bool DoPopulateRandUniform(Stream *stream,
                                     DeviceMemory<float> *v) = 0;
  virtual bool DoPopulateRandUniform(Stream *stream,
                                     DeviceMemory<double> *v) = 0;
  virtual bool DoPopulateRandUniform(Stream *stream,
                                     DeviceMemory<std::complex<float>> *v) = 0;
  virtual bool DoPopulateRandUniform(Stream *stream,
                                     DeviceMemory<std::complex<double>> *v) = 0;

  // Populates a GPU memory allocation with random values sampled from a
  // Gaussian distribution with the given mean and standard deviation.
  virtual bool DoPopulateRandGaussian(Stream *stream, float mean, float stddev,
                                      DeviceMemory<float> *v) {
    LOG(ERROR)
        << "platform's random number generator does not support gaussian";
    return false;
  }
  virtual bool DoPopulateRandGaussian(Stream *stream, double mean,
                                      double stddev, DeviceMemory<double> *v) {
    LOG(ERROR)
        << "platform's random number generator does not support gaussian";
    return false;
  }

  // Specifies the seed used to initialize the RNG.
  // This call does not transfer ownership of the buffer seed; its data should
  // not be altered for the lifetime of this call. At least 16 bytes of seed
  // data must be provided, but not all seed data will necessarily be used.
  // seed: Pointer to seed data. Must not be null.
  // seed_bytes: Size of seed buffer in bytes. Must be >= 16.
  virtual bool SetSeed(Stream *stream, const uint8 *seed,
                       uint64 seed_bytes) = 0;

 protected:
  static bool CheckSeed(const uint8 *seed, uint64 seed_bytes);
};

}  // namespace rng
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_RNG_H_
