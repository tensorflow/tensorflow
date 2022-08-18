/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_BYTE_SWAP_H_
#define TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_BYTE_SWAP_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

// Define basic byte swapping operations.
// These operations must be macros to use compiler intrinsics.
// Note that the code here is written for portability, not speed. Byte swapping
// only happens when importing a checkpoint from one hardware architecture onto
// a different architecture. If these operations become part of a fast path,
// then the function ByteSwapArray() below should be rewritten to use
// architecture-appropriate SIMD instructions that swap multiple words at once.

#if defined(__linux__)

// Use the Gnu byte swap macros when available.  See bswap(3) for more info.
#include <byteswap.h>
#define BYTE_SWAP_16(x) bswap_16(x)
#define BYTE_SWAP_32(x) bswap_32(x)
#define BYTE_SWAP_64(x) bswap_64(x)

#elif defined(PLATFORM_WINDOWS)

// On windows, byte-swapping is in winsock.h, and winsock2.h has a version of
// of htonl that can byte-swap 64-bit values.
#include <winsock2.h>
#define BYTE_SWAP_16(x) htons(x)
#define BYTE_SWAP_32(x) htonl(x)
// At the moment the 64-bit and 128-bit byte-swapping routines in Winsock2 are
// disabled in TensorFlow's standard Windows build environment, so we use
// htonl() instead of "#define BYTE_SWAP_64(x) htonll (x)".
#define BYTE_SWAP_64(x)                                \
  ((uint64_t(htonl((x)&0x00000000ffffffffUL)) << 32) | \
   (htonl(((x)&0xffffffff00000000UL) >> 32)))

#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

// On non-Linux, non-Windows, but little-endian, environments, use htonl/s,
// which byte-swap when the host byte order is little-endian. POSIX doesn't
// define a 64-bit version of these library functions, so we roll our own.
#include <arpa/inet.h>
#define BYTE_SWAP_16(x) htons(x)
#define BYTE_SWAP_32(x) htonl(x)
#define BYTE_SWAP_64(x)                                \
  ((uint64_t(htonl((x)&0x00000000ffffffffUL)) << 32) | \
   (htonl(((x)&0xffffffff00000000UL) >> 32)))

#else  // not defined(__linux__) and not defined(PLATFORM_WINDOWS)
       // and (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)

// Fall back on a non-optimized implementation on other big-endian targets.
// This code swaps one byte at a time and is probably an order of magnitude
// slower.

#define BYTE_SWAP_16(x) ((((x)&0x00ff) << 8) | (((x)&0xff00) >> 8))

#define BYTE_SWAP_32(x)                                   \
  ((((x)&0x000000ffU) << 24) | (((x)&0x0000ff00U) << 8) | \
   (((x)&0x00ff0000U) >> 8) | (((x)&0xff000000U) >> 24))

#define BYTE_SWAP_64(x)                                                      \
  ((((x)&0x00000000000000ffUL) << 56) | (((x)&0x000000000000ff00UL) << 40) | \
   (((x)&0x0000000000ff0000UL) << 24) | (((x)&0x00000000ff000000UL) << 8) |  \
   (((x)&0x000000ff00000000UL) >> 8) | (((x)&0x0000ff0000000000UL) >> 24) |  \
   (((x)&0x00ff000000000000UL) >> 40) | (((x)&0xff00000000000000UL) >> 56))

#endif  // defined(__linux__)

namespace tensorflow {

// Byte-swap an entire array of atomic C/C++ types in place.
//
// Note: When calling this function on arrays of std::complex<> types,
// multiply the number of elements by 2 and divide the bytes per element by 2.
//
// Args:
//  array: Pointer to the beginning of the array
//  bytes_per_elem: Number of bytes in each element of the array
//  array_len: Number of elements in the array
//
// Returns: Status::OK() on success, -1 otherwise
//
Status ByteSwapArray(char *array, size_t bytes_per_elem, int array_len);

// Byte-swap a tensor's backing buffer in place.
//
// Args:
//  t: Tensor to be modified IN PLACE. Any tensors that share a backing
//     buffer with this one will also end up byte-swapped.
// Returns: Status::OK() on success, -1 otherwise
// TODO(frreiss): Should this be a member of the Tensor class?
Status ByteSwapTensor(Tensor *t);

// Swap tensor_content field of Const Op Tensors in the named functions
Status ByteSwapTensorContent(MetaGraphDef *meta_graph_def);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_TENSOR_BUNDLE_BYTE_SWAP_H_
