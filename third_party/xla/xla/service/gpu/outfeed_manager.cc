/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/outfeed_manager.h"

#include <memory>

#include "absl/status/status.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

absl::Status OutfeedManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, MutableBorrowingLiteral literal) {
  ShapeTree<std::unique_ptr<gpu::OutfeedBuffer>> outfeed_buffers(
      &literal.shape());

  for (auto& leaf : outfeed_buffers.leaves()) {
    const Shape& shape = ShapeUtil::GetSubshape(literal.shape(), leaf.first);
    CHECK(shape.IsArray()) << ShapeUtil::HumanStringWithLayout(shape);
    leaf.second =
        std::make_unique<gpu::OutfeedBuffer>(ShapeUtil::ByteSizeOf(shape));
    leaf.second->set_destination(
        std::make_unique<MutableBorrowingLiteral>(literal, leaf.first));
  }

  // Give the tree of buffers to the outfeed manager. The device will fill it
  // while we're waiting for it below.
  EnqueueDestination(&outfeed_buffers);

  // Now wait till all the buffers are written.
  for (auto& leaf : outfeed_buffers.leaves()) {
    const Shape& shape = ShapeUtil::GetSubshape(literal.shape(), leaf.first);
    CHECK(shape.IsArray()) << ShapeUtil::HumanStringWithLayout(shape);
    leaf.second->WaitUntilAvailable();
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
