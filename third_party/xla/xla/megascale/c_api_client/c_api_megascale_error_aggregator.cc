/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/megascale/c_api_client/c_api_megascale_error_aggregator.h"

#include <cstddef>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/megascale/megascale_runtime_error_overlay.pb.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_megascale_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"

namespace xla::megascale::c_api_client {

CApiMegascaleErrorAggregator::ErrorDigest::~ErrorDigest() { Destroy(); }

CApiMegascaleErrorAggregator::ErrorDigest::ErrorDigest(
    ErrorDigest&& other) noexcept
    : digest_(other.digest_),
      extension_(other.extension_),
      c_api_(other.c_api_) {
  other.digest_ = nullptr;
  other.extension_ = nullptr;
  other.c_api_ = nullptr;
}

CApiMegascaleErrorAggregator::ErrorDigest&
CApiMegascaleErrorAggregator::ErrorDigest::operator=(
    ErrorDigest&& other) noexcept {
  if (this != &other) {
    Destroy();
    digest_ = other.digest_;
    extension_ = other.extension_;
    c_api_ = other.c_api_;
    other.digest_ = nullptr;
    other.extension_ = nullptr;
    other.c_api_ = nullptr;
  }
  return *this;
}

void CApiMegascaleErrorAggregator::ErrorDigest::Destroy() {
  if (digest_) {
    PJRT_Megascale_ErrorDigest_Delete_Args args{};
    args.struct_size = PJRT_Megascale_ErrorDigest_Delete_Args_STRUCT_SIZE;
    args.digest = digest_;
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error(
        extension_->error_digest_delete(&args), pjrt::MakeErrorDeleter(c_api_));
    if (error) {
      LOG(ERROR) << "Failed to delete error digest: "
                 << pjrt::PjrtErrorToStatus(error.get(), c_api_);
    }
    digest_ = nullptr;
  }
}

CApiMegascaleErrorAggregator::CApiMegascaleErrorAggregator(
    PJRT_Megascale_ErrorAggregator* aggregator,
    PJRT_Megascale_Extension* extension, const PJRT_Api* c_api)
    : aggregator_(aggregator), extension_(extension), c_api_(c_api) {}

CApiMegascaleErrorAggregator::~CApiMegascaleErrorAggregator() {
  if (aggregator_) {
    PJRT_Megascale_ErrorAggregator_Delete_Args args{};
    args.struct_size = PJRT_Megascale_ErrorAggregator_Delete_Args_STRUCT_SIZE;
    args.aggregator = aggregator_;
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error(
        extension_->error_aggregator_delete(&args),
        pjrt::MakeErrorDeleter(c_api_));
    if (error) {
      LOG(ERROR) << "Failed to delete error aggregator: "
                 << pjrt::PjrtErrorToStatus(error.get(), c_api_);
    }
  }
}

void CApiMegascaleErrorAggregator::AddError(
    absl::string_view worker_id,
    const runtime::MegaScaleRuntimeErrorOverlay& error) {
  std::string serialized_error = error.SerializeAsString();
  PJRT_Megascale_ErrorAggregator_AddError_Args args{};
  args.struct_size = PJRT_Megascale_ErrorAggregator_AddError_Args_STRUCT_SIZE;
  args.aggregator = aggregator_;
  args.worker_id = worker_id.data();
  args.worker_id_size = worker_id.size();
  args.serialized_error = serialized_error.data();
  args.serialized_error_size = serialized_error.size();
  PJRT_Error* pjrt_error = extension_->error_aggregator_add_error(&args);
  CHECK(pjrt_error == nullptr) << "Failed to add error to aggregator. "
                               << pjrt::PjrtErrorToStatus(pjrt_error, c_api_);
}

CApiMegascaleErrorAggregator::ErrorDigest
CApiMegascaleErrorAggregator::ProcessAndShutdown() {
  PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args args{};
  args.struct_size =
      PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args_STRUCT_SIZE;
  args.aggregator = aggregator_;
  PJRT_Error* error = extension_->error_aggregator_process_and_shutdown(&args);
  CHECK(error == nullptr) << "Failed to process and shutdown aggregator. "
                          << pjrt::PjrtErrorToStatus(error, c_api_);
  return ErrorDigest(args.digest, extension_, c_api_);
}

void CApiMegascaleErrorAggregator::LogErrorDigest(
    const CApiMegascaleErrorAggregator::ErrorDigest& digest) {
  PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args args{};
  args.struct_size =
      PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args_STRUCT_SIZE;
  args.aggregator = aggregator_;
  args.digest = digest.get();
  PJRT_Error* error = extension_->error_aggregator_log_error_digest(&args);
  CHECK(error == nullptr) << "Failed to log error digest. "
                          << pjrt::PjrtErrorToStatus(error, c_api_);
}

size_t CApiMegascaleErrorAggregator::size() const {
  PJRT_Megascale_ErrorAggregator_Size_Args args{};
  args.struct_size = PJRT_Megascale_ErrorAggregator_Size_Args_STRUCT_SIZE;
  args.aggregator = aggregator_;
  PJRT_Error* error = extension_->error_aggregator_size(&args);
  CHECK(error == nullptr) << "Failed to get error aggregator size. "
                          << pjrt::PjrtErrorToStatus(error, c_api_);
  return args.size;
}

bool CApiMegascaleErrorAggregator::active() const {
  PJRT_Megascale_ErrorAggregator_Active_Args args{};
  args.struct_size = PJRT_Megascale_ErrorAggregator_Active_Args_STRUCT_SIZE;
  args.aggregator = aggregator_;
  PJRT_Error* error = extension_->error_aggregator_active(&args);
  CHECK(error == nullptr) << "Failed to get error aggregator active status. "
                          << pjrt::PjrtErrorToStatus(error, c_api_);
  return args.active;
}

}  // namespace xla::megascale::c_api_client
