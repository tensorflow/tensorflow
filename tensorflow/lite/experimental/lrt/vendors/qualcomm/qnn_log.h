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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_QNN_LOG_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_QNN_LOG_H_

#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/QnnLog.h"
#include "third_party/qairt/include/QNN/System/QnnSystemInterface.h"

namespace lrt::qnn {

//
// Standalone Dump/Log Funcitonality
//

// Prints details about this interface.
void DumpInterface(const QnnInterface_t* interface);

// Prints details about this system interface.
void DumpSystemInterface(const QnnSystemInterface_t* interface);

//
// QNN SDK Usage
//

// Gets a default logger implementation to stdout.
// This is used when initializing qnn logging.
QnnLog_Callback_t GetDefaultStdOutLogger();

}  // namespace lrt::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_QNN_LOG_H_
