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

#ifndef XLA_MEGASCALE_C_API_CLIENT_C_API_MEGASCALE_ERROR_AGGREGATOR_H_
#define XLA_MEGASCALE_C_API_CLIENT_C_API_MEGASCALE_ERROR_AGGREGATOR_H_

#include <cstddef>

#include "absl/strings/string_view.h"
#include "xla/megascale/megascale_runtime_error_overlay.pb.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_megascale_extension.h"

namespace xla::megascale::c_api_client {

class CApiMegascaleErrorAggregator {
 public:
  class ErrorDigest {
   public:
    ErrorDigest(PJRT_Megascale_ErrorDigest* digest,
                PJRT_Megascale_Extension* extension, const PJRT_Api* c_api)
        : digest_(digest), extension_(extension), c_api_(c_api) {}

    ~ErrorDigest();

    ErrorDigest(ErrorDigest&& other) noexcept;
    ErrorDigest& operator=(ErrorDigest&& other) noexcept;

    PJRT_Megascale_ErrorDigest* get() const { return digest_; }

   private:
    void Destroy();

    PJRT_Megascale_ErrorDigest* digest_;
    PJRT_Megascale_Extension* extension_;
    const PJRT_Api* c_api_;
  };

  CApiMegascaleErrorAggregator(PJRT_Megascale_ErrorAggregator* aggregator,
                               PJRT_Megascale_Extension* extension,
                               const PJRT_Api* c_api);
  ~CApiMegascaleErrorAggregator();

  CApiMegascaleErrorAggregator(const CApiMegascaleErrorAggregator&) = delete;
  CApiMegascaleErrorAggregator& operator=(const CApiMegascaleErrorAggregator&) =
      delete;

  void AddError(absl::string_view worker_id,
                const runtime::MegaScaleRuntimeErrorOverlay& error);

  ErrorDigest ProcessAndShutdown();

  void LogErrorDigest(const ErrorDigest& digest);

  size_t size() const;
  bool active() const;

 private:
  PJRT_Megascale_ErrorAggregator* aggregator_;
  PJRT_Megascale_Extension* extension_;
  const PJRT_Api* c_api_;
};

}  // namespace xla::megascale::c_api_client

#endif  // XLA_MEGASCALE_C_API_CLIENT_C_API_MEGASCALE_ERROR_AGGREGATOR_H_
