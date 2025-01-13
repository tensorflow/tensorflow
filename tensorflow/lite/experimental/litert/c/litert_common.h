// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Declares canonical opaque type.
#define LITERT_DEFINE_HANDLE(name) typedef struct name##T* name

#if __ANDROID_API__ >= 26
#define LITERT_HAS_AHWB_SUPPORT 1
#else
#define LITERT_HAS_AHWB_SUPPORT 0
#endif  // __ANDROID_API__ >= 26

#if defined(__linux__) || defined(__ANDROID__)
#define LITERT_HAS_SYNC_FENCE_SUPPORT 1
#else
#define LITERT_HAS_SYNC_FENCE_SUPPORT 0
#endif

#if defined(__ANDROID__)
#define LITERT_HAS_ION_SUPPORT 1
#define LITERT_HAS_DMABUF_SUPPORT 1
#define LITERT_HAS_FASTRPC_SUPPORT 1
#else
#define LITERT_HAS_ION_SUPPORT 0
#define LITERT_HAS_DMABUF_SUPPORT 0
#define LITERT_HAS_FASTRPC_SUPPORT 0
#endif

#define LITERT_API_VERSION_MAJOR 0
#define LITERT_API_VERSION_MINOR 1
#define LITERT_API_VERSION_PATCH 0

typedef struct LiteRtApiVersion {
  int major;
  int minor;
  int patch;
} LiteRtApiVersion;

typedef enum {
  kLiteRtStatusOk = 0,

  // Generic errors.
  kLiteRtStatusErrorInvalidArgument = 1,
  kLiteRtStatusErrorMemoryAllocationFailure = 2,
  kLiteRtStatusErrorRuntimeFailure = 3,
  kLiteRtStatusErrorMissingInputTensor = 4,
  kLiteRtStatusErrorUnsupported = 5,
  kLiteRtStatusErrorNotFound = 6,
  kLiteRtStatusErrorTimeoutExpired = 7,

  // File and loading related errors.
  kLiteRtStatusErrorFileIO = 500,
  kLiteRtStatusErrorInvalidFlatbuffer = 501,
  kLiteRtStatusErrorDynamicLoading = 502,
  kLiteRtStatusErrorSerialization = 503,
  kLiteRtStatusErrorCompilation = 504,

  // IR related errors.
  kLiteRtStatusErrorIndexOOB = 1000,
  kLiteRtStatusErrorInvalidIrType = 1001,
  kLiteRtStatusErrorInvalidGraphInvariant = 1002,
  kLiteRtStatusErrorGraphModification = 1003,

  // Tool related errors.
  kLiteRtStatusErrorInvalidToolConfig = 1500,

  // Lealization related errors.
  kLiteRtStatusLegalizeNoMatch = 2000,
  kLiteRtStatusErrorInvalidLegalization = 2001,
} LiteRtStatus;

typedef enum : int {
  kLiteRtHwAccelatorNone = 0,
  kLiteRtHwAccelatorCpu = 1 << 0,
  kLiteRtHwAccelatorGpu = 1 << 1,
  kLiteRtHwAccelatorNpu = 1 << 2,
} LiteRtHwAccelerators;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMMON_H_
