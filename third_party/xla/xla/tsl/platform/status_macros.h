/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TSL_PLATFORM_STATUS_MACROS_H_
#define XLA_TSL_PLATFORM_STATUS_MACROS_H_

#include <memory>
#include <sstream>

#include "absl/status/status.h"
#include "xla/tsl/platform/statusor.h"

namespace tsl {
class StatusBuilder {
 public:
  explicit StatusBuilder(absl::Status status) : status_(std::move(status)) {}

  StatusBuilder(const StatusBuilder& other) : status_(other.status_) {
    if (other.stream_) {
      stream_ = std::make_unique<std::ostringstream>();
      stream_->str(other.stream_->str());
    }
  }

  StatusBuilder(StatusBuilder&&) = default;
  StatusBuilder& operator=(const StatusBuilder& other) {
    if (this != &other) {
      status_ = other.status_;
      if (other.stream_) {
        stream_ = std::make_unique<std::ostringstream>();
        stream_->str(other.stream_->str());
      } else {
        stream_.reset();
      }
    }
    return *this;
  }
  StatusBuilder& operator=(StatusBuilder&&) = default;

  template <typename T>
  StatusBuilder& operator<<(const T& value) & {
    if (status_.ok()) return *this;
    if (stream_ == nullptr) {
      stream_ = std::make_unique<std::ostringstream>();
      if (!status_.message().empty()) {
        *stream_ << status_.message() << "; ";
      }
    }
    *stream_ << value;
    return *this;
  }

  template <typename T>
  StatusBuilder&& operator<<(const T& value) && {
    *this << value;
    return std::move(*this);
  }

  operator absl::Status() const& { return GetStatus(); }

  operator absl::Status() && { return GetStatus(); }

 private:
  absl::Status GetStatus() const {
    if (status_.ok() || stream_ == nullptr) {
      return status_;
    }
    absl::Status new_status(status_.code(), stream_->str());
    status_.ForEachPayload(
        [&new_status](absl::string_view key, const absl::Cord& value) {
          new_status.SetPayload(key, value);
        });
    return new_status;
  }

  absl::Status status_;
  std::unique_ptr<std::ostringstream> stream_;
};
}  // namespace tsl

#ifndef ASSIGN_OR_RETURN
#define ASSIGN_OR_RETURN(lhs, rexpr) \
  ASSIGN_OR_RETURN_IMPL(             \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                          \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {         \
    return statusor.status();                       \
  }                                                 \
  lhs = std::move(statusor).value()
#endif  // ASSIGN_OR_RETURN

// TF_STATUS_MACROS_IMPL_ELSE_BLOCKER is used to prevent the "dangling else"
// problem.
#define TF_STATUS_MACROS_IMPL_ELSE_BLOCKER \
  switch (0)                               \
  case 0:                                  \
  default:

#ifndef RETURN_IF_ERROR
#define RETURN_IF_ERROR(expr)                                   \
  TF_STATUS_MACROS_IMPL_ELSE_BLOCKER                            \
  if (auto _status = (expr); ABSL_PREDICT_FALSE(!_status.ok())) \
  return tsl::StatusBuilder(std::move(_status))
#endif  // RETURN_IF_ERROR

#endif  // XLA_TSL_PLATFORM_STATUS_MACROS_H_
