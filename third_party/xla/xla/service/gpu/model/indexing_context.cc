/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/indexing_context.h"

#include <utility>

#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

static RTVarID rt_var_count = 0;

RTVar IndexingContext::RegisterRTVar(RTVarData rt_var_data) {
  rt_vars_registry_.insert(std::make_pair(rt_var_count, rt_var_data));
  return RTVar{rt_var_count++};
}

RTVarData& IndexingContext::GetRTVarData(RTVarID id) {
  return rt_vars_registry_.at(id);
}

/*static*/ void IndexingContext::ResetRTVarStateForTests() { rt_var_count = 0; }

}  // namespace gpu
}  // namespace xla
