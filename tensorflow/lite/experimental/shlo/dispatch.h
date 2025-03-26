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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_DISPATCH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_DISPATCH_H_

#define RETURN_OK_STATUS_IF_VOID(expr)                           \
  {                                                              \
    return [&](auto v) {                                         \
      if constexpr (std::is_same_v<decltype(v, (expr)), void>) { \
        (void)(expr);                                            \
        return absl::OkStatus();                                 \
      } else {                                                   \
        return expr;                                             \
      }                                                          \
      return absl::OkStatus();                                   \
    }(0);                                                        \
  }

#define DISPATCH_INT(name, element_type, ...)                           \
  {                                                                     \
    switch (element_type) {                                             \
      case DataType::kSI4:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI4>(__VA_ARGS__)));  \
      case DataType::kSI8:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI8>(__VA_ARGS__)));  \
      case DataType::kSI16:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI16>(__VA_ARGS__))); \
      case DataType::kSI32:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI32>(__VA_ARGS__))); \
      default:                                                          \
        return absl::InvalidArgumentError("Unsupported element type");  \
    }                                                                   \
  }

#define DISPATCH_FLOAT(name, element_type, ...)                         \
  {                                                                     \
    switch (element_type) {                                             \
      case DataType::kBF16:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kBF16>(__VA_ARGS__))); \
      case DataType::kF16:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kF16>(__VA_ARGS__)));  \
      case DataType::kF32:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kF32>(__VA_ARGS__)));  \
      default:                                                          \
        return absl::InvalidArgumentError("Unsupported element type");  \
    }                                                                   \
  }

#define DISPATCH_INT_FLOAT(name, element_type, ...)                     \
  {                                                                     \
    switch (element_type) {                                             \
      case DataType::kSI4:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI4>(__VA_ARGS__)));  \
      case DataType::kSI8:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI8>(__VA_ARGS__)));  \
      case DataType::kSI16:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI16>(__VA_ARGS__))); \
      case DataType::kSI32:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI32>(__VA_ARGS__))); \
      case DataType::kBF16:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kBF16>(__VA_ARGS__))); \
      case DataType::kF16:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kF16>(__VA_ARGS__)));  \
      case DataType::kF32:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kF32>(__VA_ARGS__)));  \
      default:                                                          \
        return absl::InvalidArgumentError("Unsupported element type");  \
    }                                                                   \
  }

#define DISPATCH_BOOL_INT(name, element_type, ...)                      \
  {                                                                     \
    switch (element_type) {                                             \
      case DataType::kI1:                                               \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kI1>(__VA_ARGS__)));   \
      case DataType::kSI4:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI4>(__VA_ARGS__)));  \
      case DataType::kSI8:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI8>(__VA_ARGS__)));  \
      case DataType::kSI16:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI16>(__VA_ARGS__))); \
      case DataType::kSI32:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI32>(__VA_ARGS__))); \
      default:                                                          \
        return absl::InvalidArgumentError("Unsupported element type");  \
    }                                                                   \
  }

#define DISPATCH_BOOL_INT_FLOAT(name, element_type, ...)                \
  {                                                                     \
    switch (element_type) {                                             \
      case DataType::kI1:                                               \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kI1>(__VA_ARGS__)));   \
      case DataType::kSI4:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI4>(__VA_ARGS__)));  \
      case DataType::kSI8:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI8>(__VA_ARGS__)));  \
      case DataType::kSI16:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI16>(__VA_ARGS__))); \
      case DataType::kSI32:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kSI32>(__VA_ARGS__))); \
      case DataType::kBF16:                                             \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kBF16>(__VA_ARGS__))); \
      case DataType::kF16:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kF16>(__VA_ARGS__)));  \
      case DataType::kF32:                                              \
        RETURN_OK_STATUS_IF_VOID((name<DataType::kF32>(__VA_ARGS__)));  \
      default:                                                          \
        return absl::InvalidArgumentError("Unsupported element type");  \
    }                                                                   \
  }

#define DISPATCH_QUANTIZED(name, storage_type, expressed_type, ...)          \
  {                                                                          \
    switch (storage_type) {                                                  \
      case DataType::kSI4:                                                   \
        switch (expressed_type) {                                            \
          case DataType::kBF16:                                              \
            RETURN_OK_STATUS_IF_VOID(                                        \
                (name<DataType::kSI4, DataType::kBF16>(__VA_ARGS__)));       \
          case DataType::kF16:                                               \
            RETURN_OK_STATUS_IF_VOID(                                        \
                (name<DataType::kSI4, DataType::kF16>(__VA_ARGS__)));        \
          case DataType::kF32:                                               \
            RETURN_OK_STATUS_IF_VOID(                                        \
                (name<DataType::kSI4, DataType::kF32>(__VA_ARGS__)));        \
          default:                                                           \
            return absl::InvalidArgumentError("Unsupported expressed type"); \
        }                                                                    \
        break;                                                               \
      case DataType::kSI8:                                                   \
        switch (expressed_type) {                                            \
          case DataType::kBF16:                                              \
            RETURN_OK_STATUS_IF_VOID(                                        \
                (name<DataType::kSI8, DataType::kBF16>(__VA_ARGS__)));       \
          case DataType::kF16:                                               \
            RETURN_OK_STATUS_IF_VOID(                                        \
                (name<DataType::kSI8, DataType::kF16>(__VA_ARGS__)));        \
          case DataType::kF32:                                               \
            RETURN_OK_STATUS_IF_VOID(                                        \
                (name<DataType::kSI8, DataType::kF32>(__VA_ARGS__)));        \
          default:                                                           \
            return absl::InvalidArgumentError("Unsupported expressed type"); \
        }                                                                    \
        break;                                                               \
      case DataType::kSI16:                                                  \
        switch (expressed_type) {                                            \
          case DataType::kF32:                                               \
            RETURN_OK_STATUS_IF_VOID(                                        \
                (name<DataType::kSI16, DataType::kF32>(__VA_ARGS__)));       \
          default:                                                           \
            return absl::InvalidArgumentError("Unsupported expressed type"); \
        }                                                                    \
        break;                                                               \
      default:                                                               \
        return absl::InvalidArgumentError("Unsupported storage type");       \
    }                                                                        \
  }

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_DISPATCH_H_
