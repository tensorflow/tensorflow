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

#ifndef XLA_PYTHON_IFRT_PROXY_CONTRIB_PATHWAYS_STATUS_ANNOTATOR_UTIL_H_
#define XLA_PYTHON_IFRT_PROXY_CONTRIB_PATHWAYS_STATUS_ANNOTATOR_UTIL_H_

#include "absl/status/status.h"
#include "xla/python/ifrt_proxy/contrib/pathways/status_annotator.pb.h"

namespace ifrt_proxy_contrib_pathways {

// Attaches the given `object_store_dump` to the given `status` as a payload.
void AnnotateIfrtUserStatusWithObjectStoreDump(
    absl::Status& status, const ObjectStoreDumpProto& object_store_dump);

}  // namespace ifrt_proxy_contrib_pathways

#endif  // XLA_PYTHON_IFRT_PROXY_CONTRIB_PATHWAYS_STATUS_ANNOTATOR_UTIL_H_
