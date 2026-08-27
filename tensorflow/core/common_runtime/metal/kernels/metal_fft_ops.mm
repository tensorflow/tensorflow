/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_shader_library.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// The Fourier transforms.
//
// Metal has no transform of its own, so this is a shader: one transform per
// thread along one axis, radix two when the length allows it and the direct
// sum when it does not. A multi-dimensional transform is that shader run once
// per axis, which is what separability means; the axis is addressed by a
// stride, so no transpose is needed between passes.
//
// The real forms are the same complex transform with two extra passes that
// move between the packed real layout and the complex one. Cropping the
// spectrum on the way out and mirroring it on the way in is the whole of the
// difference, because a real signal's transform is conjugate-symmetric.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct FftOp {
  int axes = 1;
  bool inverse = false;
  bool real = false;
  // The ND forms name their axes and lengths in inputs rather than in the
  // op's name.
  bool axes_from_input = false;
};

void* FftOp_Create(TF_OpKernelConstruction* ctx) { return new FftOp(); }

void FftOp_Delete(void* kernel) { delete static_cast<FftOp*>(kernel); }

bool ZeroTensor(SP_Stream stream, TF_Tensor* tensor, TF_Status* status) {
  BufferSlice slice;
  if (!SliceForTensor(tensor, &slice, status)) return false;
  const size_t bytes = TF_TensorByteSize(tensor);
  if (bytes == 0) return true;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer to zero a tensor.");
    return false;
  }
  id<MTLBlitCommandEncoder> encoder =
      [command_buffer.get() blitCommandEncoder];
  [encoder fillBuffer:slice.buffer
                range:NSMakeRange(slice.offset, bytes)
                value:0];
  [encoder endEncoding];
  command_buffer.Commit();
  return true;
}

bool Dispatch(SP_Stream stream, const char* function,
              const std::vector<BufferSlice>& buffers,
              const FftParams& params, uint32_t threads, TF_Status* status) {
  if (threads == 0) return true;
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), function, status);
  if (pipeline == nil) return false;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a transform.");
    return false;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  NSUInteger index = 0;
  for (const BufferSlice& slice : buffers) {
    [encoder setBuffer:slice.buffer offset:slice.offset atIndex:index];
    ++index;
  }
  [encoder setBytes:&params length:sizeof(params) atIndex:index];
  Dispatch1D(encoder, pipeline, threads);
  [encoder endEncoding];
  command_buffer.Commit();
  return true;
}

// Runs the transform along the last `axes` axes of a tensor whose shape is
// `shape`. `scale_inverse` divides by each transformed length, which the
// inverse transform does and the forward one does not.
bool TransformAxes(TF_OpKernelContext* ctx, SP_Stream stream,
                   const BufferSlice& data, const std::vector<int64_t>& shape,
                   int axes, bool inverse, TF_Status* status) {
  const int rank = static_cast<int>(shape.size());
  int64_t longest = 1;
  for (int a = rank - axes; a < rank; ++a) {
    longest = std::max(longest, shape[a]);
  }
  int64_t widest = 1;
  for (int a = rank - axes; a < rank; ++a) {
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < a; ++i) outer *= shape[i];
    for (int i = a + 1; i < rank; ++i) inner *= shape[i];
    widest = std::max(widest, outer * inner);
  }
  // Two working copies of the longest axis for every transform in flight.
  const std::vector<int64_t> scratch_shape = {widest * 2 * longest * 2};
  ScopedTensor scratch;
  scratch.reset(TF_AllocateTemp(ctx, TF_FLOAT, scratch_shape.data(), 1,
                                nullptr, status));
  if (TF_GetCode(status) != TF_OK) return false;
  BufferSlice scratch_slice;
  if (!SliceForTensor(scratch.get(), &scratch_slice, status)) return false;

  // Innermost axis first, which is only a matter of taste: the passes commute.
  for (int a = rank - 1; a >= rank - axes; --a) {
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < a; ++i) outer *= shape[i];
    for (int i = a + 1; i < rank; ++i) inner *= shape[i];
    FftParams params;
    params.outer = static_cast<uint32_t>(outer);
    params.n = static_cast<uint32_t>(shape[a]);
    params.inner = static_cast<uint32_t>(inner);
    params.count = static_cast<uint32_t>(outer * inner);
    params.inverse = inverse ? 1 : 0;
    params.scale = inverse ? 1 : 0;
    params.padding0 = 0;
    params.padding1 = 0;
    std::vector<BufferSlice> buffers = {data, scratch_slice};
    if (!Dispatch(stream, "tf_fft_axis_float", buffers, params, params.count,
                  status)) {
      return false;
    }
  }
  return true;
}

// The requested transform length, which crops or zero-pads the input.
bool ReadFftLength(TF_OpKernelContext* ctx, int axes,
                   std::vector<int64_t>* out, TF_Status* status) {
  ScopedTensor length;
  TF_GetInput(ctx, 1, length.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  const void* data = TF_TensorData(length.get());
  if (data == nullptr || TF_TensorElementCount(length.get()) != axes) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: fft_length must have one entry per transformed axis, "
                 "in host memory.");
    return false;
  }
  out->clear();
  for (int i = 0; i < axes; ++i) {
    out->push_back(static_cast<const int32_t*>(data)[i]);
    if (out->back() <= 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: fft_length must be positive.");
      return false;
    }
  }
  return true;
}

// Moves a tensor into a different shape of the same rank, cropping what does
// not fit and padding the rest with zeros. `mode` also converts between the
// real and complex layouts, which is what the real transforms need on the way
// in and on the way out.
bool ResizeInto(SP_Stream stream, const BufferSlice& input,
                const std::vector<int64_t>& in_shape,
                const BufferSlice& output,
                const std::vector<int64_t>& out_shape, uint32_t mode,
                TF_Status* status) {
  if (in_shape.size() != out_shape.size() || in_shape.size() > 8) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: a transform of this rank is not supported.");
    return false;
  }
  ResizeParams params;
  params.rank = static_cast<uint32_t>(in_shape.size());
  params.count = static_cast<uint32_t>(ElementCount(out_shape));
  params.mode = mode;
  params.padding0 = 0;
  for (int i = 0; i < 8; ++i) {
    params.in_shape[i] = 1;
    params.out_shape[i] = 1;
  }
  for (size_t i = 0; i < in_shape.size(); ++i) {
    params.in_shape[i] = static_cast<uint32_t>(in_shape[i]);
    params.out_shape[i] = static_cast<uint32_t>(out_shape[i]);
  }
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), "tf_fft_resize_float", status);
  if (pipeline == nil) return false;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a transform.");
    return false;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input.buffer offset:input.offset atIndex:0];
  [encoder setBuffer:output.buffer offset:output.offset atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
  return true;
}

// The complex forms: the output has the input's shape and the transform runs
// in place on a copy of it.
void ComplexFft_ComputeImpl(FftOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int rank = static_cast<int>(in_shape.size());
  if (rank < op->axes) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the input has fewer axes than the transform.");
    return;
  }
  // The fixed-rank forms transform the input as it stands; the ND forms are
  // told what length to use and crop or pad to it.
  std::vector<int64_t> shape = in_shape;
  if (op->axes_from_input) {
    std::vector<int64_t> fft_length;
    if (!ReadFftLength(ctx, op->axes, &fft_length, status)) return;
    for (int i = 0; i < op->axes; ++i) {
      shape[rank - op->axes + i] = fft_length[i];
    }
  }
  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_COMPLEX64, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * sizeof(float) * 2, status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  BufferSlice in_slice, out_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  if (shape == in_shape) {
    OrderedCommandBuffer command_buffer(stream);
    if (!command_buffer.ok()) {
      TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                   "Metal: could not create a command buffer for a "
                   "transform.");
      return;
    }
    id<MTLBlitCommandEncoder> encoder =
        [command_buffer.get() blitCommandEncoder];
    [encoder copyFromBuffer:in_slice.buffer
               sourceOffset:in_slice.offset
                   toBuffer:out_slice.buffer
          destinationOffset:out_slice.offset
                       size:static_cast<NSUInteger>(count) * sizeof(float) * 2];
    [encoder endEncoding];
    command_buffer.Commit();
  } else if (!ResizeInto(stream, in_slice, in_shape, out_slice, shape, 0,
                         status)) {
    return;
  }
  TransformAxes(ctx, stream, out_slice, shape, op->axes, op->inverse, status);
}

// The real transforms, over any number of trailing axes.
//
// Only the innermost axis is halved: a real signal's transform is
// conjugate-symmetric along the axis it was real in, and the outer axes carry
// full complex spectra. So the forward direction pads or crops every
// transformed axis, transforms them all, and keeps half of the innermost; the
// inverse completes the innermost half by symmetry, transforms, and keeps the
// real part.
void RealFft_ComputeImpl(FftOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int rank = static_cast<int>(in_shape.size());
  if (rank < op->axes) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the input has fewer axes than the transform.");
    return;
  }
  std::vector<int64_t> fft_length;
  if (!ReadFftLength(ctx, op->axes, &fft_length, status)) return;
  const int64_t last = fft_length.back();
  const int64_t half = last / 2 + 1;

  // The shape the complex transform runs on, and the shape it reports.
  std::vector<int64_t> work_shape = in_shape;
  std::vector<int64_t> out_shape = in_shape;
  for (int i = 0; i < op->axes; ++i) {
    const int axis = rank - op->axes + i;
    work_shape[axis] = fft_length[i];
    out_shape[axis] = fft_length[i];
  }
  out_shape[rank - 1] = op->inverse ? last : half;

  const TF_DataType out_dtype = op->inverse ? TF_FLOAT : TF_COMPLEX64;
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, out_dtype, out_shape.data(), rank,
      static_cast<size_t>(ElementCount(out_shape)) *
          TF_DataTypeSize(out_dtype),
      status));
  if (TF_GetCode(status) != TF_OK) return;
  if (ElementCount(out_shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  BufferSlice in_slice, out_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  const std::vector<int64_t> work_bytes = {ElementCount(work_shape) * 2};
  ScopedTensor work;
  work.reset(
      TF_AllocateTemp(ctx, TF_FLOAT, work_bytes.data(), 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return;
  BufferSlice work_slice;
  if (!SliceForTensor(work.get(), &work_slice, status)) return;

  if (op->inverse) {
    // The half spectrum has to be completed along the innermost axis before
    // anything else, and the outer axes are cropped or padded as they are.
    std::vector<int64_t> mirror_in = in_shape;
    for (int i = 0; i + 1 < op->axes; ++i) {
      mirror_in[rank - op->axes + i] = fft_length[i];
    }
    // Everything but the innermost axis first, so the mirror sees rows of the
    // right length.
    std::vector<int64_t> staged_shape = mirror_in;
    ScopedTensor staged;
    BufferSlice staged_slice = in_slice;
    if (staged_shape != in_shape) {
      const std::vector<int64_t> staged_bytes = {ElementCount(staged_shape) *
                                                 2};
      staged.reset(TF_AllocateTemp(ctx, TF_FLOAT, staged_bytes.data(), 1,
                                   nullptr, status));
      if (TF_GetCode(status) != TF_OK) return;
      if (!SliceForTensor(staged.get(), &staged_slice, status)) return;
      if (!ResizeInto(stream, in_slice, in_shape, staged_slice, staged_shape,
                      0, status)) {
        return;
      }
    }
    int64_t rows = 1;
    for (int i = 0; i + 1 < rank; ++i) rows *= staged_shape[i];
    FftParams params;
    params.outer = static_cast<uint32_t>(rows);
    params.n = static_cast<uint32_t>(last);
    params.inner = static_cast<uint32_t>(staged_shape[rank - 1]);
    params.count = static_cast<uint32_t>(rows);
    params.inverse = 1;
    params.scale = 1;
    params.padding0 = 0;
    params.padding1 = 0;
    std::vector<BufferSlice> mirror = {staged_slice, work_slice};
    if (!Dispatch(stream, "tf_fft_mirror_spectrum_float", mirror, params,
                  static_cast<uint32_t>(rows * last), status)) {
      return;
    }
  } else {
    if (!ResizeInto(stream, in_slice, in_shape, work_slice, work_shape, 1,
                    status)) {
      return;
    }
  }

  if (!TransformAxes(ctx, stream, work_slice, work_shape, op->axes,
                     op->inverse, status)) {
    return;
  }

  // On the way out, keep half the innermost axis, or its real part.
  ResizeInto(stream, work_slice, work_shape, out_slice, out_shape,
             op->inverse ? 2 : 0, status);
}

// The ND forms take the axes to transform and their lengths as inputs. Both
// are read on the host, since the axes decide which passes run and the lengths
// decide the output's shape.
//
// The axes handled are the trailing ones in increasing order, which is what
// every caller of these ops passes and what the two- and three-dimensional
// forms are by definition. A different set would need the tensor transposed
// first, which is work this does not do rather than work it does badly.
bool ReadAxes(TF_OpKernelContext* ctx, int rank, int* axes,
              TF_Status* status) {
  ScopedTensor t;
  TF_GetInput(ctx, 2, t.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  const void* data = TF_TensorData(t.get());
  const int64_t count = TF_TensorElementCount(t.get());
  if (data == nullptr || count < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: axes must be a non-empty vector in host memory.");
    return false;
  }
  std::vector<int> requested;
  for (int64_t i = 0; i < count; ++i) {
    int axis = static_cast<const int32_t*>(data)[i];
    if (axis < 0) axis += rank;
    requested.push_back(axis);
  }
  for (size_t i = 0; i < requested.size(); ++i) {
    const int expected = rank - static_cast<int>(requested.size()) +
                         static_cast<int>(i);
    if (requested[i] != expected) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: the N-dimensional transforms handle the trailing "
                   "axes in increasing order.");
      return false;
    }
  }
  *axes = static_cast<int>(requested.size());
  return true;
}

void FftNd_ComputeImpl(FftOp* op, TF_OpKernelContext* ctx, bool real,
                       bool inverse, TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const int rank = static_cast<int>(ShapeOf(input.get()).size());
  int axes = 1;
  if (!ReadAxes(ctx, rank, &axes, status)) return;
  op->axes = axes;
  op->inverse = inverse;
  op->real = real;
  op->axes_from_input = true;
  if (real) {
    RealFft_ComputeImpl(op, ctx, status);
  } else {
    // The complex ND form is told its lengths too, and crops or pads to them
    // before transforming, which the fixed-rank forms never have to do.
    ComplexFft_ComputeImpl(op, ctx, status);
  }
}

#define METAL_FFT_COMPUTE(NAME, AXES, INVERSE, REAL)                        \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<FftOp*>(kernel);                                 \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a transform kernel has no state.");              \
    } else {                                                                \
      op->axes = AXES;                                                      \
      op->inverse = INVERSE;                                                \
      op->real = REAL;                                                      \
      if (REAL) {                                                           \
        RealFft_ComputeImpl(op, ctx, status);                               \
      } else {                                                              \
        ComplexFft_ComputeImpl(op, ctx, status);                            \
      }                                                                     \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_FFT_COMPUTE(Fft1_Compute, 1, false, false)
METAL_FFT_COMPUTE(Fft2_Compute, 2, false, false)
METAL_FFT_COMPUTE(Fft3_Compute, 3, false, false)
METAL_FFT_COMPUTE(Ifft1_Compute, 1, true, false)
METAL_FFT_COMPUTE(Ifft2_Compute, 2, true, false)
METAL_FFT_COMPUTE(Ifft3_Compute, 3, true, false)
void FftNd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<FftOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: FFTND has no state.");
  } else {
    FftNd_ComputeImpl(op, ctx, /*real=*/false, /*inverse=*/false, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void IfftNd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<FftOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: IFFTND has no state.");
  } else {
    FftNd_ComputeImpl(op, ctx, /*real=*/false, /*inverse=*/true, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void RfftNd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<FftOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: RFFTND has no state.");
  } else {
    FftNd_ComputeImpl(op, ctx, /*real=*/true, /*inverse=*/false, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void IrfftNd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<FftOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: IRFFTND has no state.");
  } else {
    FftNd_ComputeImpl(op, ctx, /*real=*/true, /*inverse=*/true, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

METAL_FFT_COMPUTE(Rfft1_Compute, 1, false, true)
METAL_FFT_COMPUTE(Rfft2_Compute, 2, false, true)
METAL_FFT_COMPUTE(Rfft3_Compute, 3, false, true)
METAL_FFT_COMPUTE(Irfft1_Compute, 1, true, true)
METAL_FFT_COMPUTE(Irfft2_Compute, 2, true, true)
METAL_FFT_COMPUTE(Irfft3_Compute, 3, true, true)

#undef METAL_FFT_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name, bool host_length,
              bool host_axes = false) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &FftOp_Create, compute, &FftOp_Delete);
  // The transform length sizes the output, so it is read on the host, and so
  // are the axes when the op names them.
  if (host_length) TF_KernelBuilder_HostMemory(builder, "fft_length");
  if (host_axes) TF_KernelBuilder_HostMemory(builder, "axes");
  TF_RegisterKernelBuilder(name.c_str(), builder, status);
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalFftKernels() {
  Register("FFT", &Fft1_Compute, "MetalFFT", false);
  Register("FFT2D", &Fft2_Compute, "MetalFFT2D", false);
  Register("FFT3D", &Fft3_Compute, "MetalFFT3D", false);
  Register("IFFT", &Ifft1_Compute, "MetalIFFT", false);
  Register("IFFT2D", &Ifft2_Compute, "MetalIFFT2D", false);
  Register("IFFT3D", &Ifft3_Compute, "MetalIFFT3D", false);
  // The deprecated spellings are the same ops under their old names.
  Register("BatchFFT", &Fft1_Compute, "MetalBatchFFT", false);
  Register("BatchFFT2D", &Fft2_Compute, "MetalBatchFFT2D", false);
  Register("BatchFFT3D", &Fft3_Compute, "MetalBatchFFT3D", false);
  Register("BatchIFFT", &Ifft1_Compute, "MetalBatchIFFT", false);
  Register("BatchIFFT2D", &Ifft2_Compute, "MetalBatchIFFT2D", false);
  Register("BatchIFFT3D", &Ifft3_Compute, "MetalBatchIFFT3D", false);
  Register("RFFT", &Rfft1_Compute, "MetalRFFT", true);
  Register("RFFT2D", &Rfft2_Compute, "MetalRFFT2D", true);
  Register("RFFT3D", &Rfft3_Compute, "MetalRFFT3D", true);
  Register("IRFFT", &Irfft1_Compute, "MetalIRFFT", true);
  Register("IRFFT2D", &Irfft2_Compute, "MetalIRFFT2D", true);
  Register("IRFFT3D", &Irfft3_Compute, "MetalIRFFT3D", true);
  // The axes are read on the host as well: they decide which passes run.
  Register("FFTND", &FftNd_Compute, "MetalFFTND", true, true);
  Register("IFFTND", &IfftNd_Compute, "MetalIFFTND", true, true);
  Register("RFFTND", &RfftNd_Compute, "MetalRFFTND", true, true);
  Register("IRFFTND", &IrfftNd_Compute, "MetalIRFFTND", true, true);
}

}  // namespace metal
}  // namespace tensorflow
