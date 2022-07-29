/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/core/profiler/rpc/client/tensorboard_export.h"

#include <iostream>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/convert/xplane_to_profile_response.h"
#include "tensorflow/core/profiler/rpc/client/save_profile.h"
#include "tensorflow/core/profiler/rpc/client/populate_request.h"

namespace tensorflow {
namespace profiler {

Status ExportToTensorBoard(const XSpace& xspace, const std::string& logdir) {
  TF_RETURN_IF_ERROR(MaybeCreateEmptyEventFile(logdir));

  ProfileResponse response;
  ProfileRequest request = PopulateProfileRequest(
      GetTensorBoardProfilePluginDir(logdir), GetCurrentTimeStampAsString(),
      port::Hostname(), /*options=*/{});
  TF_RETURN_IF_ERROR(
      ConvertXSpaceToProfileResponse(xspace, request, &response));
  std::stringstream ss;  // Record LOG messages.
  TF_RETURN_IF_ERROR(SaveProfile(request.repository_root(),
                                 request.session_id(), request.host_name(),
                                 response, &ss));
  LOG(INFO) << ss.str();
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
