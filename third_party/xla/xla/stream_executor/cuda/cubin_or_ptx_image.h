/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUBIN_OR_PTX_IMAGE_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUBIN_OR_PTX_IMAGE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace stream_executor {
// This is the input to various PTX compilation and linking functions. The
// struct holds either PTX or CUBIN in `bytes` and a compilation profile in
// `profile`. `profile` can either be a compile profile `compute_XY` or a SASS
// profile `sm_XY`.
struct CubinOrPTXImage {
  std::string profile;
  std::vector<uint8_t> bytes;
};
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUBIN_OR_PTX_IMAGE_H_
