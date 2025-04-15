/* Copyright 2015 The OpenXLA Authors.

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
// Note that this interface is optionally supported by platforms..
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
//    .ThenFft(plan.get(), x, &y);
//  TF_CHECK_OK(stream.BlockHostUntilDone());
//
// By using stream operations in this manner the user can easily intermix custom
// kernel launches with these pre-canned FFT routines.

#ifndef XLA_STREAM_EXECUTOR_FFT_H_
#define XLA_STREAM_EXECUTOR_FFT_H_

#include <complex>
#include <cstdint>
#include <memory>

namespace stream_executor {

class Stream;
template <typename ElemT>
class DeviceMemory;
class ScratchAllocator;

namespace fft {

// Specifies FFT input and output types, and the direction.
// R, D, C, and Z stand for SP real, DP real, SP complex, and DP complex.
enum class Type {
  kInvalid,
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

  // Creates a batched FFT plan with scratch allocator.
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
  virtual std::unique_ptr<Plan> CreateBatchedPlanWithScratchAllocator(
      Stream *stream, int rank, uint64_t *elem_count, uint64_t *input_embed,
      uint64_t input_stride, uint64_t input_distance, uint64_t *output_embed,
      uint64_t output_stride, uint64_t output_distance, Type type,
      bool in_place_fft, int batch_count,
      ScratchAllocator *scratch_allocator) = 0;

  // Updates the plan's work area with space allocated by a new scratch
  // allocator. This facilitates plan reuse with scratch allocators.
  //
  // This requires that the plan was originally created using a scratch
  // allocator, as otherwise scratch space will have been allocated internally
  // by cuFFT.
  virtual void UpdatePlanWithScratchAllocator(
      Stream *stream, Plan *plan, ScratchAllocator *scratch_allocator) = 0;

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
  FftSupport(const FftSupport &) = delete;
  void operator=(const FftSupport &) = delete;
};

// Macro used to quickly declare overrides for abstract virtuals in the
// fft::FftSupport base class. Assumes that it's emitted somewhere inside the
// ::stream_executor namespace.
#define TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES                   \
  std::unique_ptr<fft::Plan> CreateBatchedPlanWithScratchAllocator(            \
      Stream *stream, int rank, uint64_t *elem_count, uint64_t *input_embed,   \
      uint64_t input_stride, uint64_t input_distance, uint64_t *output_embed,  \
      uint64_t output_stride, uint64_t output_distance, fft::Type type,        \
      bool in_place_fft, int batch_count, ScratchAllocator *scratch_allocator) \
      override;                                                                \
  void UpdatePlanWithScratchAllocator(Stream *stream, fft::Plan *plan,         \
                                      ScratchAllocator *scratch_allocator)     \
      override;                                                                \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<std::complex<float>> &input,                   \
             DeviceMemory<std::complex<float>> *output) override;              \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<std::complex<double>> &input,                  \
             DeviceMemory<std::complex<double>> *output) override;             \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<float> &input,                                 \
             DeviceMemory<std::complex<float>> *output) override;              \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<double> &input,                                \
             DeviceMemory<std::complex<double>> *output) override;             \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<std::complex<float>> &input,                   \
             DeviceMemory<float> *output) override;                            \
  bool DoFft(Stream *stream, fft::Plan *plan,                                  \
             const DeviceMemory<std::complex<double>> &input,                  \
             DeviceMemory<double> *output) override;

}  // namespace fft
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_FFT_H_
