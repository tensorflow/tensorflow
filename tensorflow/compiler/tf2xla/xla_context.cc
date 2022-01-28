/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_context.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

const char XlaContext::kXlaContextResourceName[] = "_xla_context";

// Looks up the context associated with the current step. It is stored
// in a resource container managed by the device.
/* static */ XlaContext& XlaContext::Get(const OpKernelContext* ctx) {
  // When an Op kernel wants to use an XLA JIT context, the
  // per-step context is looked up in the resource manager. The
  // JIT will prepopulate the JITContext.
  XlaContext* context;
  TF_CHECK_OK(ctx->step_container()->Lookup(ctx->resource_manager(),
                                            kXlaContextResourceName, &context));
  // The resource manager handed us a fresh reference to 'context', but retains
  // a reference itself so the context won't be freed. The resource manager will
  // outlive the JIT compilation.
  context->Unref();
  return *context;
}

void XlaContext::set_args(std::vector<XlaExpression> args) {
  args_ = std::move(args);
}

XlaContext::XlaContext(XlaCompiler* compiler, xla::XlaBuilder* builder,
                       const Graph* graph)
    : compiler_(compiler), builder_(builder) {
  if (graph) {
    for (const Node* node : graph->nodes()) {
      stack_traces_[node->name()] = node->GetStackTrace();
    }
  }
}

string XlaContext::DebugString() const { return "XLA JIT context"; }

void XlaContext::SetRetval(int index, const XlaExpression& expression) {
  const int64_t retvals_size = retvals_.size();
  if (retvals_size <= index) {
    retvals_.resize(index + 1);
  }
  retvals_[index] = expression;
}

XlaResource* XlaContext::AddResource(std::unique_ptr<XlaResource> resource) {
  resources_.push_back(std::move(resource));
  return resources_.back().get();
}

const xla::XlaComputation* XlaContext::GetOrCreateMax(const DataType type) {
  return LookupOrCreate(type, &max_func_, [type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Max() for " << type_string;
    xla::XlaBuilder b("max<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x =
        xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y =
        xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    xla::Max(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::XlaComputation* XlaContext::GetOrCreateMin(const DataType type) {
  return LookupOrCreate(type, &min_func_, [type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Min() for " << type_string;
    xla::XlaBuilder b("min<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x =
        xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y =
        xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    xla::Min(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::XlaComputation* XlaContext::GetOrCreateAdd(const DataType type) {
  return LookupOrCreate(type, &add_func_, [type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Add() for " << type_string;
    xla::XlaBuilder b("add<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x =
        xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y =
        xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    xla::Add(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::XlaComputation* XlaContext::GetOrCreateMul(const DataType type) {
  return LookupOrCreate(type, &mul_func_, [type] {
    const string type_string = DataTypeString(type);
    VLOG(1) << "Building Mul() for " << type_string;
    xla::XlaBuilder b("mul<" + type_string + ">");
    xla::PrimitiveType xla_type;
    TF_CHECK_OK(DataTypeToPrimitiveType(type, &xla_type));
    auto x =
        xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla_type, {}), "x");
    auto y =
        xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla_type, {}), "y");
    xla::Mul(x, y);
    return b.Build().ConsumeValueOrDie();
  });
}

const xla::XlaComputation* XlaContext::LookupOrCreate(
    DataType type, ComputationMap* out,
    const std::function<xla::XlaComputation()>& create) {
  {
    const auto& entry = (*out)[type];
    if (!entry.IsNull()) {
      return &entry;
    }
  }
  auto new_entry = create();
  {
    // Somebody else might have made one concurrently.
    auto& entry = (*out)[type];
    if (entry.IsNull()) {
      entry = std::move(new_entry);
    }
    return &entry;
  }
}

Status XlaContext::RecordCollectiveInfoFromNestedCompilationResult(
    const XlaCompilationResult& result) {
  if (result.collective_info) {
    return RecordCollectiveInfo(result.collective_info->group_key,
                                result.collective_info->group_size)
        .status();
  }
  return Status::OK();
}

StatusOr<int64_t> XlaContext::RecordCollectiveInfo(int group_key,
                                                   int group_size) {
  if (!collective_info_) {
    collective_info_ = {group_key, group_size, 0};
  } else if (collective_info_->group_key != group_key ||
             collective_info_->group_size != group_size) {
    return errors::InvalidArgument(
        "Only single configuration of CollectiveReduceV2Op is ",
        "supported in a given cluster. Recorded group_key=",
        collective_info_->group_key,
        " attempting to insert group_key=", group_key);
  }

  // Create the channel_id to be used for the collective. Avoid having the
  // same channel_id to be used for 2 or more collectives since XLA attempts
  // to "gang schedule" all collectives with the same channel_id.
  return (static_cast<int64_t>(group_key) << 32) | collective_info_->next_id++;
}

}  // namespace tensorflow
