/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_DISPATCH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_DISPATCH_H_

namespace stablehlo {

#define DISPATCH_INT(name, element_type, ...)                             \
  {                                                                       \
    switch (element_type) {                                               \
      case ElementType::kSI8:                                             \
        return name<ElementType::kSI8, ElementType::kSI8>(__VA_ARGS__);   \
      case ElementType::kSI16:                                            \
        return name<ElementType::kSI16, ElementType::kSI16>(__VA_ARGS__); \
      case ElementType::kSI32:                                            \
        return name<ElementType::kSI32, ElementType::kSI32>(__VA_ARGS__); \
      default:                                                            \
        return absl::InvalidArgumentError("Unsupported element type");    \
    }                                                                     \
  }

#define DISPATCH_FLOAT(name, element_type, ...)                           \
  {                                                                       \
    switch (element_type) {                                               \
      case ElementType::kBF16:                                            \
        return name<ElementType::kBF16, ElementType::kBF16>(__VA_ARGS__); \
      case ElementType::kF16:                                             \
        return name<ElementType::kF16, ElementType::kF16>(__VA_ARGS__);   \
      case ElementType::kF32:                                             \
        return name<ElementType::kF32, ElementType::kF32>(__VA_ARGS__);   \
      default:                                                            \
        return absl::InvalidArgumentError("Unsupported element type");    \
    }                                                                     \
  }

#define DISPATCH_INT_FLOAT(name, element_type, ...)                       \
  {                                                                       \
    switch (element_type) {                                               \
      case ElementType::kSI8:                                             \
        return name<ElementType::kSI8, ElementType::kSI8>(__VA_ARGS__);   \
      case ElementType::kSI16:                                            \
        return name<ElementType::kSI16, ElementType::kSI16>(__VA_ARGS__); \
      case ElementType::kSI32:                                            \
        return name<ElementType::kSI32, ElementType::kSI32>(__VA_ARGS__); \
      case ElementType::kBF16:                                            \
        return name<ElementType::kBF16, ElementType::kBF16>(__VA_ARGS__); \
      case ElementType::kF16:                                             \
        return name<ElementType::kF16, ElementType::kF16>(__VA_ARGS__);   \
      case ElementType::kF32:                                             \
        return name<ElementType::kF32, ElementType::kF32>(__VA_ARGS__);   \
      default:                                                            \
        return absl::InvalidArgumentError("Unsupported element type");    \
    }                                                                     \
  }

#define DISPATCH_BOOL_INT_FLOAT(name, element_type, ...)                  \
  {                                                                       \
    switch (element_type) {                                               \
      case ElementType::kI1:                                              \
        return name<ElementType::kI1, ElementType::kI1>(__VA_ARGS__);     \
      case ElementType::kSI8:                                             \
        return name<ElementType::kSI8, ElementType::kSI8>(__VA_ARGS__);   \
      case ElementType::kSI16:                                            \
        return name<ElementType::kSI16, ElementType::kSI16>(__VA_ARGS__); \
      case ElementType::kSI32:                                            \
        return name<ElementType::kSI32, ElementType::kSI32>(__VA_ARGS__); \
      case ElementType::kBF16:                                            \
        return name<ElementType::kBF16, ElementType::kBF16>(__VA_ARGS__); \
      case ElementType::kF16:                                             \
        return name<ElementType::kF16, ElementType::kF16>(__VA_ARGS__);   \
      case ElementType::kF32:                                             \
        return name<ElementType::kF32, ElementType::kF32>(__VA_ARGS__);   \
      default:                                                            \
        return absl::InvalidArgumentError("Unsupported element type");    \
    }                                                                     \
  }

#define DISPATCH_QUANTIZED(name, storage_type, expressed_type, ...)           \
  {                                                                           \
    switch (storage_type) {                                                   \
      case ElementType::kSI8:                                                 \
        switch (expressed_type) {                                             \
          case ElementType::kBF16:                                            \
            return name<ElementType::kSI8, ElementType::kBF16>(__VA_ARGS__);  \
          case ElementType::kF16:                                             \
            return name<ElementType::kSI8, ElementType::kF16>(__VA_ARGS__);   \
          case ElementType::kF32:                                             \
            return name<ElementType::kSI8, ElementType::kF32>(__VA_ARGS__);   \
          default:                                                            \
            return absl::InvalidArgumentError("Unsupported expressed type");  \
        }                                                                     \
        break;                                                                \
      case ElementType::kSI16:                                                \
        switch (expressed_type) {                                             \
          case ElementType::kBF16:                                            \
            return name<ElementType::kSI16, ElementType::kBF16>(__VA_ARGS__); \
          case ElementType::kF16:                                             \
            return name<ElementType::kSI16, ElementType::kF16>(__VA_ARGS__);  \
          case ElementType::kF32:                                             \
            return name<ElementType::kSI16, ElementType::kF32>(__VA_ARGS__);  \
          default:                                                            \
            return absl::InvalidArgumentError("Unsupported expressed type");  \
        }                                                                     \
        break;                                                                \
      case ElementType::kSI32:                                                \
        switch (expressed_type) {                                             \
          case ElementType::kBF16:                                            \
            return name<ElementType::kSI32, ElementType::kBF16>(__VA_ARGS__); \
          case ElementType::kF16:                                             \
            return name<ElementType::kSI32, ElementType::kF16>(__VA_ARGS__);  \
          case ElementType::kF32:                                             \
            return name<ElementType::kSI32, ElementType::kF32>(__VA_ARGS__);  \
          default:                                                            \
            return absl::InvalidArgumentError("Unsupported expressed type");  \
        }                                                                     \
        break;                                                                \
      default:                                                                \
        return absl::InvalidArgumentError("Unsupported storage type");        \
    }                                                                         \
  }

}  // namespace stablehlo

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_LEGACY_SRC_DISPATCH_H_
