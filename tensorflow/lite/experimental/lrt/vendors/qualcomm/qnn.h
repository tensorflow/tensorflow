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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_QNN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_QNN_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/QnnLog.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "third_party/qairt/include/QNN/System/QnnSystemInterface.h"

namespace lrt {
namespace qnn {

class Qnn {
 public:
  Qnn(const Qnn&) = delete;
  Qnn(Qnn&&) = delete;
  Qnn& operator=(const Qnn&) = delete;
  Qnn& operator=(Qnn&&) = delete;

  ~Qnn();

  static absl::StatusOr<std::unique_ptr<Qnn>> Create();

  const Qnn_LogHandle_t& log_handle() const { return log_handle_; }
  const QNN_SYSTEM_INTERFACE_VER_TYPE& system_interface() const {
    return system_interface_;
  }
  const QNN_INTERFACE_VER_TYPE& qnn_interface() const { return qnn_interface_; }

 private:
  static constexpr const char* kSystemLibPath = "libQnnSystem.so";
  static constexpr const char* kQnnHtpLibPath = "libQnnHtp.so";

  Qnn() = default;

  absl::Status LoadSystemSymbols();
  absl::Status LoadQnnSymbols();

  void* system_dlib_handle_ = nullptr;
  void* qnn_dlib_handle_ = nullptr;
  QNN_SYSTEM_INTERFACE_VER_TYPE system_interface_;
  QNN_INTERFACE_VER_TYPE qnn_interface_;
  Qnn_LogHandle_t log_handle_ = nullptr;
};

}  // namespace qnn
}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_QNN_H_
