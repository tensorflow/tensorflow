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

#ifndef XLA_STREAM_EXECUTOR_CUDA_PTX_LINKING_METHOD_H_
#define XLA_STREAM_EXECUTOR_CUDA_PTX_LINKING_METHOD_H_

#include <ostream>

#include "absl/strings/str_cat.h"

namespace stream_executor {

enum class PtxLinkingMethod {
  kNone,
  kNvLink,
  kDriver,
};

template <typename Sink>
void AbslStringify(Sink& sink, const PtxLinkingMethod& method) {
  switch (method) {
    case PtxLinkingMethod::kNvLink:
      sink.Append("NvLink");
      break;
    case PtxLinkingMethod::kDriver:
      sink.Append("Driver");
      break;
    case PtxLinkingMethod::kNone:
      sink.Append("None");
      break;
  }
}

inline std::ostream& operator<<(std::ostream& os,
                                const PtxLinkingMethod& method) {
  return os << absl::StrCat(method);
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_PTX_LINKING_METHOD_H_
