/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_STATUS_H_
#define TENSORFLOW_CORE_PLATFORM_STATUS_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/status.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::FromAbslStatus;
using tsl::OkStatus;
using tsl::Status;
using tsl::StatusCallback;
using tsl::StatusGroup;
using tsl::TfCheckOpHelper;
using tsl::TfCheckOpHelperOutOfLine;
using tsl::ToAbslStatus;

namespace errors {
typedef tsl::errors::Code Code;
using tsl::errors::GetStackTrace;
using tsl::errors::SetStackTrace;
}  // namespace errors
// NOLINTEND(misc-unused-using-decls)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STATUS_H_
