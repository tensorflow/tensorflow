/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_USE_SYCL
#error This file contains SYCL specific code and should only be included when \
building with SYCL support.
#endif

#ifndef TENSORFLOW_CORE_UTIL_SYCL_UTIL_H_
#define TENSORFLOW_CORE_UTIL_SYCL_UTIL_H_

#define ConvertToGlobalTypeSycl(Scalar, dev_pointer)             \
  static_cast<typename cl::sycl::global_ptr<Scalar>::pointer_t>( \
      static_cast<void*>((dev_pointer)))

namespace tensorflow {
template <class T>
struct SYCLDevicePointer {
#if defined(__SYCL_DEVICE_ONLY__)
  typedef typename cl::sycl::global_ptr<T>::pointer_t PointerType;
#else
  typedef T* PointerType;
#endif
};

// Need an atomic add for the MaxPoolGrad and MaxPool3DGrad kernels.
//
// For the device, this needs a pointer to global memory which isn't understood
// by the host. The host should never be calling this method, but we provide
// the header so that the host compiler can compile the functor.

template <typename T>
inline void SyclAtomicAdd(typename SYCLDevicePointer<T>::PointerType address,
                   const T increment);

// Use the OpenCL atomic uint operations to provide a floating point atomic add.
// For the device we use the atomic compare-exchange builtin to keep trying to
// add to the memory in a thread safe way. The union is needed as these
// builtins are not availble for floating point types, only integer types, so
// we do the addition on the float and the memory update on the uint.
//
// TODO(jwlawson): Remove once we have different type accessors for SYCL buffers
// Providing a way to cast the types of buffers or accessors has been proposed
// as a SYCL extension, so once this is available we can use an atomic
// accessor and remove this.
#ifdef __SYCL_DEVICE_ONLY__
template <>
inline void SyclAtomicAdd<float>(
    typename SYCLDevicePointer<float>::PointerType address,
    const float increment) {
  union {
    uint32_t u32;
    float f32;
  } next, expected, current;
  current.f32 = *address;
  auto uint_addr = ConvertToGlobalTypeSycl(uint32_t, address);
  do {
    expected.f32 = current.f32;
    next.f32 = expected.f32 + increment;
    current.u32 =
        _Z14atomic_cmpxchgPVU3AS1jjj(uint_addr, expected.u32, next.u32);
  } while (current.u32 != expected.u32);
}
template <>
inline void SyclAtomicAdd<double>(
    typename SYCLDevicePointer<double>::PointerType address,
    const double increment) {
  union {
    uint64_t u64;
    double d64;
  } next, expected, current;
  current.d64 = *address;
  auto uint_addr = ConvertToGlobalTypeSycl(uint64_t, address);
  do {
    expected.d64 = current.d64;
    next.d64 = expected.d64 + increment;
    current.d64 = _Z12atom_cmpxchgPVU3AS1mmm(uint_addr, expected.u64, next.u64);
  } while (current.u64 != expected.u64);
}
#else
// Provide a dummy implementation for the host compiler. This code will not be
// seen by the SYCL device, and so should not be run.
template <>
inline void SyclAtomicAdd<float>(float* address, const float increment) {
  LOG(FATAL) << "MaxPool3DGradSYCL should only be run on a SYCL device";
}
template <>
inline void SyclAtomicAdd<double>(double* address, const double increment) {
  LOG(FATAL) << "MaxPool3DGradSYCL should only be run on a SYCL device";
}
#endif  // __SYCL_DEVICE_ONLY__

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_UTIL_SYCL_UTIL_H_
