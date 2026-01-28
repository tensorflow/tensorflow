/* Copyright 2020 The JAX Authors

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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/pjrt/status_casters.h"
#include "xla/service/call_inliner.h"

namespace nb = nanobind;

namespace xla {
namespace {

NB_MODULE(_hlo_pass, m) {
  // Hlo Module Passes
  nb::class_<HloPassInterface> hlo_pass_interface(m, "HloPassInterface");
  hlo_pass_interface.def_prop_ro("name", &HloPassInterface::name)
      .def("is_pass_pipeline", &HloPassInterface::IsPassPipeline)
      .def("run", [](HloPassInterface& pass, HloModule* module) -> bool {
        return xla::ValueOrThrow(pass.Run(module));
      });

  nb::class_<HloDCE, HloPassInterface>(m, "HloDCE").def(nb::init<>());
  nb::class_<CallInliner, HloPassInterface>(m, "CallInliner").def(nb::init<>());
  nb::class_<FlattenCallGraph, HloPassInterface>(m, "FlattenCallGraph")
      .def(nb::init<>());
  nb::class_<TupleSimplifier, HloPassInterface>(m, "TupleSimplifier")
      .def(nb::init<>());
}

}  // namespace
}  // namespace xla
