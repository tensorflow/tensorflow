/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_API_DLSYM_SET_FN_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_API_DLSYM_SET_FN_H_

#define TFTPU_SET_FN(Struct, FnName)                                       \
  Struct->FnName##Fn =                                                     \
      reinterpret_cast<decltype(FnName)*>(dlsym(library_handle, #FnName)); \
  if (!(Struct->FnName##Fn)) {                                             \
    LOG(FATAL) << #FnName " not available in this library.";               \
    return absl::UnimplementedError(#FnName                                \
                                    " not available in this library.");    \
  }

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_API_DLSYM_SET_FN_H_
