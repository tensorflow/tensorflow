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

#include "xla/pjrt/pjrt_build_stamp_internal.h"

#include <cstdint>
#include <string>

#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"

namespace xla {

void SetMosaicVersion(PjRtBuildStampProto& proto, int64_t major_version,
                      int64_t minor_version) {
  for (int i = 0; i < proto.versions_size(); ++i) {
    if (proto.versions(i).name() == kBuildStampVersionMosaic) {
      proto.mutable_versions(i)->set_major_version(major_version);
      proto.mutable_versions(i)->set_minor_version(minor_version);
      return;
    }
  }
  auto* ver = proto.add_versions();
  ver->set_name(kBuildStampVersionMosaic);
  ver->set_major_version(major_version);
  ver->set_minor_version(minor_version);
}

}  // namespace xla
