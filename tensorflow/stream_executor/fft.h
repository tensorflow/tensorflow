/* Copyright 2015 Google Inc. All Rights Reserved.

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

// Exposes the family of FFT routines as pre-canned high performance calls for
// use in conjunction with the StreamExecutor abstraction.
//
// Note that this interface is optionally supported by platforms; see
// StreamExecutor::SupportsFft() for details.
//
// This abstraction makes it simple to entrain FFT operations on GPU data into
// a Stream -- users typically will not use this API directly, but will use the
// Stream builder methods to entrain these operations "under the hood". For
// example:
//
//  DeviceMemory<std::complex<float>> x =
//    stream_exec->AllocateArray<std::complex<float>>(1024);
//  DeviceMemory<std::complex<float>> y =
//    stream_exec->AllocateArray<std::complex<float>>(1024);
//  // ... populate x and y ...
//  Stream stream{stream_exec};
//  std::unique_ptr<Plan> plan =
//     stream_exec.AsFft()->Create1dPlan(&stream, 1024, Type::kC2CForward);
//  stream
//    .Init()
//    .ThenFft(plan.get(), x, &y)
//    .BlockHostUntilDone();
//
// By using stream operations in this manner the user can easily intermix custom
// kernel launches (via StreamExecutor::ThenLaunch()) with these pre-canned FFT
// routines.

#ifndef TENSORFLOW_STREAM_EXECUTOR_FFT_H_
#define TENSORFLOW_STREAM_EXECUTOR_FFT_H_

#include <complex>
#include <memory>
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

class Stream;
template <typename ElemT>
class DeviceMemory;

namespace fft {

// Specifies FFT input and output types, and the direction.
// R, D, C, and Z stand for SP real, DP real, SP complex, and DP complex.
enum class Type {
  kC2CForward,
  kC2CInverse,
  kC2R,
  kR2C,
  kZ2ZForward,
  kZ2ZInverse,
  kZ2D,
  kD2Z
};

// FFT plan class. Each FFT implementation should define a plan class that is
// derived from this class. It does not provide any interface but serves
// as a common type that is used to execute the plan.
class Plan {
 public:
  virtual ~Plan() {}
};

// FFT support interface -- this can be derived from a GPU executor when the
// underlying platform has an FFT library implementation available. See
// StreamExecutor::AsFft().
//
// This support interface is not generally thread-safe; it is only thread-safe
// for the CUDA platform (cuFFT) usage; host side FFT support is known
// thread-compatible, but not thread-safe.
class FftSupport {
 public:
  virtual ~FftSupport() {}

  // Creates a 1d FFT plan.
  virtual std::unique_ptr<Plan> Create1dPlan(Stream *stream, uint64 num_x,
                                             Type type, bool in_place_fft) = 0;

  // Creates a 2d FFT plan.
  virtual std::unique_ptr<Plan> Create2dPlan(Stream *stream, uint64 num_x,
                                             uint64 num_y, Type type,
                                             bool in_place_fft) = 0;

  // Creates a 3d FFT plan.
  virtual std::unique_ptr<Plan> Create3dPlan(Stream *stream, uint64 num_x,
                                             uint64 num_y, uint64 num_z,
                                             Type type, bool in_place_fft) = 0;

  // Creates a batched FFT plan.
  //
  // stream:          The GPU stream in which the FFT runs.
  // rank:            Dimensionality of the transform (1, 2, or 3).
  // elem_count:      Array of size rank, describing the size of each dimension.
  // input_embed, output_embed:
  //                  Pointer of size rank that indicates the storage dimensions
  //                  of the input/output data in memory. If set to null_ptr all
  //                  other advanced data layout parameters are ignored.
  // input_stride:    Indicates the distance (number of elements; same below)
  //                  between two successive input elements.
  // input_distance:  Indicates the distance between the first element of two
  //                  consecutive signals in a batch of the input data.
  // output_stride:   Indicates the distance between two successive output
  //                  elements.
  // output_distance: Indicates the distance between the first element of two
  //                  consecutive signals in a batch of the output data.
  virtual std::unique_ptr<Plan> CreateBatchedPlan(
      Stream *stream, int rank, uint64 *elem_count, uint64 *input_embed,
      uint64 input_stride, uint64 input_distance, uint64 *output_embed,
      uint64 output_stride, uint64 output_distance, Type type,
      bool in_place_fft, int batch_count) = 0;

  // Computes complex-to-complex FFT in the transform direction as specified
  // by direction parameter.
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<std::complex<float>> &input,
                     DeviceMemory<std::complex<float>> *output) = 0;
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<std::complex<double>> &input,
                     DeviceMemory<std::complex<double>> *output) = 0;

  // Computes real-to-complex FFT in forward direction.
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<float> &input,
                     DeviceMemory<std::complex<float>> *output) = 0;
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<double> &input,
                     DeviceMemory<std::complex<double>> *output) = 0;

  // Computes complex-to-real FFT in inverse direction.
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<std::complex<float>> &input,
                     DeviceMemory<float> *output) = 0;
  virtual bool DoFft(Stream *stream, Plan *plan,
                     const DeviceMemory<std::complex<double>> &input,
                     DeviceMemory<double> *output) = 0;

 protected:
  FftSupport() {}

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(FftSupport);
};

// Macro used to quickly declare overrides for abstract virtuals in the
// fft::FftSupport base class. Assumes that it's emitted somewhere inside the
// ::perftools::gputools namespace.
#define TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES                \
  std::unique_ptr<fft::Plan> Create1dPlan(Stream *stream, uint64 num_x,      \
                                          fft::Type type, bool in_place_fft) \
      override;                                                              \
  std::unique_ptr<fft::Plan> Create2dPlan(Stream *stream, uint64 num_x,      \
                                          uint64 num_y, fft::Type type,      \
                                          bool in_place_fft) override;       \
  std::unique_ptr<fft::Plan> Create3dPlan(                                   \
      Stream *stream, uint64 num_x, uint64 num_y, uint64 num_z,              \
      fft::Type type, bool in_place_fft) override;                           \
  std::unique_ptr<fft::Plan> CreateBatchedPlan(                              \
      Stream *stream, int rank, uint64 *elem_count, uint64 *input_embed,     \
      uint64 input_stride, uint64 input_distance, uint64 *output_embed,      \
      uint64 output_stride, uint64 output_distance, fft::Type type,          \
      bool in_place_fft, int batch_count) override;                          \
  bool DoFft(Stream *stream, fft::Plan *plan,                                \
             const DeviceMemory<std::complex<float>> &input,                 \
             DeviceMemory<std::complex<float>> *output) override;            \
  bool DoFft(Stream *stream, fft::Plan *plan,                                \
             const DeviceMemory<std::complex<double>> &input,                \
             DeviceMemory<std::complex<double>> *output) override;           \
  bool DoFft(Stream *stream, fft::Plan *plan,                                \
             const DeviceMemory<float> &input,                               \
             DeviceMemory<std::complex<float>> *output) override;            \
  bool DoFft(Stream *stream, fft::Plan *plan,                                \
             const DeviceMemory<double> &input,                              \
             DeviceMemory<std::complex<double>> *output) override;           \
  bool DoFft(Stream *stream, fft::Plan *plan,                                \
             const DeviceMemory<std::complex<float>> &input,                 \
             DeviceMemory<float> *output) override;                          \
  bool DoFft(Stream *stream, fft::Plan *plan,                                \
             const DeviceMemory<std::complex<double>> &input,                \
             DeviceMemory<double> *output) override;

}  // namespace fft
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_FFT_H_
