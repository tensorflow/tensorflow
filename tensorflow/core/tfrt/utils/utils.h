/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_H_
#define TENSORFLOW_CORE_TFRT_UTILS_H_

#include <string>

#include "absl/status/status.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
class Device;
class EagerContext;
}  // namespace tensorflow

namespace tfrt {

class BEFFile;
class ExecutionContext;
class HostContext;

typedef tensorflow::gtl::InlinedVector<tfrt::DType, 4> TfrtDataTypeVector;
typedef tensorflow::gtl::ArraySlice<tfrt::DType> TfrtDataTypeSlice;

// TODO(b/161370736): Have a formal method to convert between TF's and TFRT's
// device name. Currently TFRT adopts the suffix of TF's device name,
// e.g. CPU:0.
Expected<const char*> ConvertTfDeviceNameToTfrt(
    const char* device_name, tensorflow::EagerContext* eager_context);

DType ConvertTfDTypeToTfrtDType(tensorflow::DataType dtype);

// Runs the runtime initialization function. A runtime initialization function
// is added by runtime/compiler workflow and is not present in the original
// savedmodel.
//
// TODO(b/178714905): We should avoid special handling on initialization by
// letting compiler to handle it.
tensorflow::Status RunRuntimeInitializer(const tfrt::ExecutionContext& exec_ctx,
                                         tfrt::BEFFile* bef_file,
                                         absl::string_view fallback_init_func);

// Creates dummy TF devices from the input device names. Currently this method
// is used to create the TPU_SYSTEM device for worker server.
void CreateDummyTfDevices(
    const std::vector<std::string>& device_names,
    std::vector<std::unique_ptr<tensorflow::Device>>* dummy_tf_devices);

// Creates and add dummy TFRT devices from the input device names. Currently
// this method is used to create the TPU_SYSTEM device for worker server.
void AddDummyTfrtDevices(const std::vector<std::string>& device_names,
                         tfrt::HostContext* host_ctx);

// Creates a BEF file from a BEF buffer. `runtime` is used to provide host
// context for opening `bef`.
tensorflow::StatusOr<RCReference<tfrt::BEFFile>> CreateBefFileFromBefBuffer(
    const tensorflow::tfrt_stub::Runtime& runtime, const tfrt::BefBuffer& bef);

// Returns a unique integer within this process.
int64_t GetUniqueInt();

// A list of macros similar to `TF_RETURN_IF_ERROR`, with additional model
// loading stage info.
#define RETURN_IF_ERROR_IN_IMPORT(...) \
  RETURN_IF_ERROR_WITH_STAGE_INFO("GraphDef proto -> MLIR", __VA_ARGS__)

#define RETURN_IF_ERROR_IN_COMPILE(...)                                       \
  RETURN_IF_ERROR_WITH_STAGE_INFO(                                            \
      "TF dialect -> TFRT dialect, compiler issue, please contact MLIR team", \
      __VA_ARGS__)

#define RETURN_IF_ERROR_IN_INIT(...) \
  RETURN_IF_ERROR_WITH_STAGE_INFO("Initialize TFRT", __VA_ARGS__)

#define RETURN_IF_ERROR_WITH_STAGE_INFO(stage, ...)                           \
  do {                                                                        \
    ::tensorflow::Status _status = (__VA_ARGS__);                             \
    if (TF_PREDICT_FALSE(!_status.ok())) {                                    \
      return ::tensorflow::Status(_status.code(),                             \
                                  ::tensorflow::strings::StrCat(              \
                                      stage, ": ", _status.error_message())); \
    }                                                                         \
  } while (0)

// A list of macros similar to `TF_ASSIGN_OR_RETURN`, with additional model
// loading stage info.
#define ASSIGN_OR_RETURN_IN_IMPORT(lhs, rexpr) \
  ASSIGN_OR_RETURN_WITH_STAGE_INFO("GraphDef proto -> MLIR", lhs, rexpr)

#define ASSIGN_OR_RETURN_IN_COMPILE(lhs, rexpr)                               \
  ASSIGN_OR_RETURN_WITH_STAGE_INFO(                                           \
      "TF dialect -> TFRT dialect, compiler issue, please contact MLIR team", \
      lhs, rexpr)

#define ASSIGN_OR_RETURN_IN_INIT(lhs, rexpr) \
  ASSIGN_OR_RETURN_WITH_STAGE_INFO("Initialize TFRT", lhs, rexpr)

#define ASSIGN_OR_RETURN_WITH_STAGE_INFO(stage, lhs, rexpr)                    \
  ASSIGN_OR_RETURN_WITH_STAGE_INFO_IMPL(                                       \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), stage, lhs, \
      rexpr)

#define ASSIGN_OR_RETURN_WITH_STAGE_INFO_IMPL(statusor, stage, lhs, rexpr)    \
  auto statusor = (rexpr);                                                    \
  if (TF_PREDICT_FALSE(!statusor.ok())) {                                     \
    const auto& _status = statusor.status();                                  \
    return ::tensorflow::Status(                                              \
        _status.code(),                                                       \
        ::tensorflow::strings::StrCat(stage, ": ", _status.error_message())); \
  }                                                                           \
  lhs = std::move(statusor.ValueOrDie())

}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_UTILS_H_
