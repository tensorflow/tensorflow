/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/collective_util.h"

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace collective_util {

/*static*/
Status InitializeDeviceAndLocality(const DeviceMgr* dev_mgr,
                                   const string& device_name, Device** device,
                                   DeviceLocality* device_locality) {
  if (!dev_mgr) {
    return errors::Internal("Required non-null dev_mgr ", dev_mgr,
                            " for InitializeDeviceAndLocality");
  }

  Status status = dev_mgr->LookupDevice(device_name, device);
  if (status.ok()) {
    CHECK(*device);
    *device_locality = (*device)->attributes().locality();
  } else {
    LOG(ERROR) << "Failed to find device " << device_name;
    for (auto d : dev_mgr->ListDevices()) {
      LOG(ERROR) << "Available devices " << d->name();
    }
  }
  return status;
}

/*static*/
string SubdivPermDebugString(const CollectiveParams& col_params) {
  const auto& subdiv_perms =
      col_params.instance.impl_details.subdiv_permutations;
  string buf;
  for (int sdi = 0; sdi < subdiv_perms.size(); ++sdi) {
    strings::StrAppend(&buf, "Subdiv ", sdi, " device order:\n");
    for (int di = 0; di < subdiv_perms[sdi].size(); ++di) {
      int idx = subdiv_perms[sdi][di];
      if (idx >= 0) {
        CHECK_GT(col_params.instance.device_names.size(), idx);
        strings::StrAppend(&buf, col_params.instance.device_names[idx], "\n");
      }
    }
    strings::StrAppend(&buf, " subdiv_offsets: ");
    for (auto o : col_params.instance.impl_details.subdiv_offsets)
      strings::StrAppend(&buf, o, " ");
    strings::StrAppend(&buf, " SubdivRank: ");
    for (auto d : col_params.subdiv_rank) strings::StrAppend(&buf, d, " ");
    if (col_params.instance.type == BROADCAST_COLLECTIVE) {
      strings::StrAppend(&buf, " subdiv_source_rank: ");
      for (auto src : col_params.instance.impl_details.subdiv_source_rank)
        strings::StrAppend(&buf, src, " ");
    }
    strings::StrAppend(&buf, "\n");
  }
  return buf;
}

}  // namespace collective_util
}  // namespace tensorflow
