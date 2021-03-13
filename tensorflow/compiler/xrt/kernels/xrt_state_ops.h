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

// Classes for allocating XLA literals in device memory and managing handles
// that refer to them.

#ifndef TENSORFLOW_COMPILER_XRT_KERNELS_XRT_STATE_OPS_H_
#define TENSORFLOW_COMPILER_XRT_KERNELS_XRT_STATE_OPS_H_

#include <functional>
#include <memory>
#include <string>

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_device.h"
#include "tensorflow/compiler/xrt/xrt_memory_manager.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/percentile_sampler.h"
#include "tensorflow/core/lib/monitoring/timed.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Helper functions for templated ops.
class XRTStateHelpers {
 public:
  // The Status return value allows us to use the
  // TF_ASSIGN_OR_RETURN macro, which doesn't work within the body of an
  // OpKernel::Compute method.
  static Status MakeLiteral(const xla::LiteralProto& proto,
                            xla::Literal* literal) {
    TF_ASSIGN_OR_RETURN(*literal, xla::Literal::CreateFromProto(proto));
    return Status::OK();
  }

  // ParseTupleNode is the recursive function used to parse a recursive
  // xrt::XLATupleNode proto and generate the xla::Shape of the 'spine' i.e. the
  // tuple shape where every leaf is an existing allocation. As a side-effect it
  // fills in input_vector by looking up allocations from handles in the
  // input_tensor_list as they are referenced by nodes in the proto.
  static Status ParseTupleNode(
      const xrt::XLATupleNode& tuple_node, const OpInputList& input_tensor_list,
      std::vector<XRTTupleAllocation::ExpandedTupleInput>* input_vector,
      xla::Shape* shape, ResourceMgr* rm) {
    if (tuple_node.tuples_size() > 0) {
      // This is an internal node in the proto so descend recursively.
      xla::Shape dummy = xla::ShapeUtil::MakeShapeWithType<float>({});
      std::vector<xla::Shape> subshapes(tuple_node.tuples_size(), dummy);
      *xla::ShapeUtil::GetMutableSubshape(shape, {}) =
          xla::ShapeUtil::MakeTupleShape(subshapes);
      for (int i = 0; i < tuple_node.tuples_size(); ++i) {
        TF_RETURN_IF_ERROR(ParseTupleNode(
            tuple_node.tuples(i), input_tensor_list, input_vector,
            xla::ShapeUtil::GetMutableSubshape(shape, {i}), rm));
      }
    } else {
      // This is a leaf node in the proto so look up the referenced input.
      int input_index = tuple_node.input_index();
      if (input_index < 0 || input_index >= input_vector->size()) {
        return errors::InvalidArgument("Invalid tuple input index ",
                                       input_index, ": MakeTuple has ",
                                       input_vector->size(), " inputs.");
      }
      bool release_this_input = tuple_node.release_input_handle();
      XRTTupleAllocation::ExpandedTupleInput& input =
          input_vector->at(input_index);
      if (input.allocation != nullptr &&
          (input.release_allocation_after_use || release_this_input)) {
        return errors::InvalidArgument(
            "Invalid tuple tree: input index ", input_index,
            " is repeated but release_input_handle is true.");
      }
      if (input.allocation == nullptr) {
        // We haven't dereferenced this handle yet.
        TF_RET_CHECK(
            TensorShapeUtils::IsScalar(input_tensor_list[input_index].shape()));
        int64 key = input_tensor_list[input_index].scalar<int64>()();
        TF_ASSIGN_OR_RETURN(input.allocation,
                            XRTMemoryManager::Get(rm)->Lookup(key));
        input.release_allocation_after_use = release_this_input;
      }
    }
    return Status::OK();
  }

  // Parses a xrt::XLATupleNode proto recursively and returns the corresponding
  // ShapeTree where each leaf is an allocation corresponding to a handle in
  // input_tensor_list. The ordinal of one of the allocations is returned in
  // device_ordinal. Since it's not possible to specify a xrt::XLATupleNode with
  // no leaves, device_ordinal will always be filled in by a successful call to
  // ParseTupleTree.
  static Status ParseTupleTree(
      const xrt::XLATupleNode& tuple_tree_root,
      const OpInputList& input_tensor_list,
      std::vector<XRTTupleAllocation::ExpandedTupleInput>* input_vector,
      xla::ShapeTree<XRTTupleAllocation::ExpandedTupleInput>* tuple_shape_tree,
      int* device_ordinal, ResourceMgr* rm) {
    // First get the shape of the 'spine' of the new tuple, where every leaf is
    // an existing allocation. As a side-effect dereference the input handles
    // into allocations in input_vector.
    xla::Shape tuple_tree_shape;
    TF_RETURN_IF_ERROR(ParseTupleNode(tuple_tree_root, input_tensor_list,
                                      input_vector, &tuple_tree_shape, rm));
    // Make the shape tree of allocations where the shape is the spine and each
    // leaf is one of the allocations looked up in input_vector. Internal nodes
    // have nullptr allocations.
    *tuple_shape_tree = xla::ShapeTree<XRTTupleAllocation::ExpandedTupleInput>(
        tuple_tree_shape);
    tuple_shape_tree->ForEachMutableElement(
        [&](const xla::ShapeIndex& index,
            XRTTupleAllocation::ExpandedTupleInput* element) {
          if (tuple_shape_tree->IsLeaf(index)) {
            // Find the matching leaf in the proto tree.
            const xrt::XLATupleNode* tuple_node = &tuple_tree_root;
            for (int i = 0; i < index.size(); ++i) {
              tuple_node = &tuple_node->tuples(index[i]);
            }
            // Copy the appropriate input allocation to the leaf of the
            // tuple_shape_tree.
            int input_index = tuple_node->input_index();
            *element = input_vector->at(input_index);
            CHECK(element->release_allocation_after_use ==
                  tuple_node->release_input_handle());
            // We just need to know the device_ordinal of one of the
            // allocations. We will validate later that they are all the same.
            *device_ordinal = (*element).allocation->device_ordinal();
          }
        });
    return Status::OK();
  }
};

// Op that allocates memory for a literal and transfers it to the device.
template <class DeviceAccessor>
class XRTAllocateOp : public OpKernel {
 public:
  explicit XRTAllocateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~XRTAllocateOp() override = default;
  XRTAllocateOp(const XRTAllocateOp&) = delete;
  XRTAllocateOp& operator=(const XRTAllocateOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTAllocateOp::Compute";
    auto timed = monitoring::MakeTimed(xrt_metrics::GetAllocateCell());

    const Tensor& allocation_info = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(allocation_info.shape()),
                errors::Internal("allocation input should be a string scalar"));
    xrt::XLAAllocation allocation_proto;
    OP_REQUIRES(ctx,
                ParseFromTString(allocation_info.scalar<tstring>()(),
                                 &allocation_proto),
                errors::InvalidArgument(
                    "Unable to parse allocation input to XLAAllocation"));

    xla::Literal literal;
    OP_REQUIRES_OK(
        ctx, XRTStateHelpers::MakeLiteral(allocation_proto.value(), &literal));

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    // We are guaranteed that the underlying device object won't be deleted out
    // from under us, while the ScopedRef is live.
    class DeviceAccessor::ScopedRef device_ref;
    OP_REQUIRES_OK(ctx, DeviceAccessor::InitScopedRef(ctx, &device_ref));

    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    XRTTupleAllocation* allocation;
    OP_REQUIRES_OK(ctx, XRTTupleAllocation::CreateAndTransfer(
                            literal, memory_manager.get(), device_ref.backend(),
                            device_ref.device_ordinal(), &allocation));

    Tensor output(DT_INT64, TensorShape({}));
    output.scalar<int64>()() = memory_manager->Register(allocation);
    ctx->set_output(0, output);
  }
};

// Op that allocates uninitialized memory on the device for a tensor of
// a particular shape.
template <class DeviceAccessor>
class XRTAllocateUninitializedOp : public OpKernel {
 public:
  explicit XRTAllocateUninitializedOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &tf_shape_));
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, tf_shape_, &xla_shape_));
  }
  ~XRTAllocateUninitializedOp() override = default;
  XRTAllocateUninitializedOp(const XRTAllocateUninitializedOp&) = delete;
  XRTAllocateUninitializedOp& operator=(const XRTAllocateUninitializedOp&) =
      delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTAllocateUninitializedOp::Compute";
    auto timed =
        monitoring::MakeTimed(xrt_metrics::GetAllocateUninitializedCell());
    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    // We are guaranteed that the underlying device object won't be deleted out
    // from under us, while the ScopedRef is live.
    class DeviceAccessor::ScopedRef device_ref;
    OP_REQUIRES_OK(ctx, DeviceAccessor::InitScopedRef(ctx, &device_ref));

    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    XRTTupleAllocation* allocation;
    OP_REQUIRES_OK(ctx,
                   XRTTupleAllocation::CreateUninitialized(
                       xla_shape_, memory_manager.get(), device_ref.backend(),
                       device_ref.device_ordinal(), &allocation));

    Tensor output(DT_INT64, TensorShape({}));
    output.scalar<int64>()() = memory_manager->Register(allocation);
    ctx->set_output(0, output);
  }

 private:
  DataType dtype_;
  TensorShape tf_shape_;
  xla::Shape xla_shape_;
};

// Op that allocates memory for a tensor (with optional layout) and transfers it
// to the device, returning an allocation handle.
template <class DeviceAccessor>
class XRTAllocateFromTensorOp : public OpKernel {
 public:
  explicit XRTAllocateFromTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    bool make_tuple = false;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &tf_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("make_tuple", &make_tuple));
    std::vector<int64> minor_to_major;
    if (ctx->HasAttr("layouts")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("layouts", &minor_to_major));
    }
    OP_REQUIRES(
        ctx, tf_shapes_.size() == dtypes_.size(),
        errors::InvalidArgument("shapes and dtypes must be the same length"));
    std::vector<xla::Shape> xla_shapes;
    xla_shapes.reserve(tf_shapes_.size());
    for (int i = 0; i < tf_shapes_.size(); i++) {
      xla::Shape xla_shape;
      OP_REQUIRES_OK(
          ctx, TensorShapeToXLAShape(dtypes_[i], tf_shapes_[i], &xla_shape));
      xla_shapes.push_back(std::move(xla_shape));
    }
    if (xla_shapes.size() > 1 || make_tuple) {
      shape_ = xla::ShapeUtil::MakeTupleShape(xla_shapes);
    } else {
      shape_.Swap(&xla_shapes.front());
    }
    if (!minor_to_major.empty()) {
      xla::Shape shape_with_layouts;
      OP_REQUIRES_OK(ctx, GetShapeWithLayout(shape_, minor_to_major,
                                             /*layout_func=*/nullptr,
                                             &shape_with_layouts));
      shape_.Swap(&shape_with_layouts);
    }
  }

  ~XRTAllocateFromTensorOp() override = default;
  XRTAllocateFromTensorOp(const XRTAllocateFromTensorOp&) = delete;
  XRTAllocateFromTensorOp& operator=(const XRTAllocateFromTensorOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTAllocateFromTensorOp::Compute";
    auto timed =
        monitoring::MakeTimed(xrt_metrics::GetAllocateFromTensorCell());

    OpInputList values;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &values));
    OP_REQUIRES(ctx, values.size() == tf_shapes_.size(),
                errors::InvalidArgument(
                    "Wrong number of inputs to XRTAllocateFromTensor: ",
                    values.size(), " vs. ", tf_shapes_.size()));

    std::vector<const char*> tensors_data;
    for (size_t i = 0; i < values.size(); ++i) {
      const Tensor& input_tensor = values[i];
      OP_REQUIRES(ctx, input_tensor.dtype() == dtypes_[i],
                  errors::InvalidArgument(
                      "Input tensor type and input dtype do not match"));
      // We allow the requested on-device shape to differ from the shape of the
      // input tensor, as long as they have the same number of elements.
      OP_REQUIRES(
          ctx,
          input_tensor.shape().num_elements() == tf_shapes_[i].num_elements(),
          errors::InvalidArgument(
              "Input tensor must have the number of elements specified "
              "in the matching input shape: ",
              input_tensor.shape().num_elements(), " vs. ",
              tf_shapes_[i].num_elements(), " at index ", i));
      tensors_data.push_back(
          static_cast<const char*>(DMAHelper::base(&input_tensor)));
    }
    // Use the buffer straight out of the input tensors to create the literal.
    xla::BorrowingLiteral literal =
        shape_.IsTuple() ? xla::BorrowingLiteral(tensors_data, shape_)
                         : xla::BorrowingLiteral(tensors_data.front(), shape_);
    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    // We are guaranteed that the underlying device object won't be deleted out
    // from under us, while the ScopedRef is live.
    class DeviceAccessor::ScopedRef device_ref;
    OP_REQUIRES_OK(ctx, DeviceAccessor::InitScopedRef(ctx, &device_ref));

    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    XRTTupleAllocation* allocation;
    OP_REQUIRES_OK(ctx, XRTTupleAllocation::CreateAndTransfer(
                            literal, memory_manager.get(), device_ref.backend(),
                            device_ref.device_ordinal(), &allocation));

    Tensor output(DT_INT64, TensorShape({}));
    output.scalar<int64>()() = memory_manager->Register(allocation);
    ctx->set_output(0, output);
  }

 private:
  std::vector<TensorShape> tf_shapes_;
  DataTypeVector dtypes_;
  xla::Shape shape_;
};

// Op that takes a tuple handle input and returns a handle to a sub-tuple of the
// input.
template <bool discard_, class DeviceAccessor>
class XRTSubTupleOp : public OpKernel {
 public:
  explicit XRTSubTupleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~XRTSubTupleOp() override = default;
  XRTSubTupleOp(const XRTSubTupleOp&) = delete;
  XRTSubTupleOp& operator=(const XRTSubTupleOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTSubTupleOp::Compute";
    auto timed = monitoring::MakeTimed(xrt_metrics::GetSubTupleCell());

    const Tensor& handle_tensor = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(handle_tensor.shape()),
        errors::Internal("computation input should be an int64 scalar"));
    int64 allocation_handle = handle_tensor.scalar<int64>()();

    const Tensor& subtuple_info = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(subtuple_info.shape()),
        errors::Internal("tuple index input should be an int32 vector"));
    xla::ShapeIndex shape_index;
    for (int i = 0; i < subtuple_info.dim_size(0); ++i) {
      shape_index.push_back(subtuple_info.vec<int32>()(i));
    }

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    RefPtr<XRTTupleAllocation> allocation;
    OP_REQUIRES_OK(ctx, memory_manager->Lookup(allocation_handle, &allocation));

    if (discard_) {
      VLOG(2) << "Releasing handle " << allocation_handle;
      OP_REQUIRES_OK(ctx, memory_manager->Release(allocation_handle));
    }

    XRTTupleAllocation* suballocation;
    OP_REQUIRES_OK(
        ctx, XRTTupleAllocation::MakeSubBuffer(allocation.get(), shape_index,
                                               &suballocation, !discard_));

    Tensor output(DT_INT64, TensorShape({}));
    output.scalar<int64>()() = memory_manager->Register(suballocation);
    ctx->set_output(0, output);
  }
};

// Op that allocates memory for a literal and transfers it to the device.
template <class DeviceAccessor>
class XRTMakeTupleOp : public OpKernel {
 public:
  explicit XRTMakeTupleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~XRTMakeTupleOp() override = default;
  XRTMakeTupleOp(const XRTMakeTupleOp&) = delete;
  XRTMakeTupleOp& operator=(const XRTMakeTupleOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTMakeTupleOp::Compute";
    auto timed = monitoring::MakeTimed(xrt_metrics::GetMakeTupleCell());

    const Tensor& tuple_info = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(tuple_info.shape()),
        errors::Internal("tuple description input should be a string scalar"));
    xrt::XLATupleNode tuple_proto;
    OP_REQUIRES(
        ctx, ParseFromTString(tuple_info.scalar<tstring>()(), &tuple_proto),
        errors::InvalidArgument("Unable to parse tuple input to XLATupleNode"));

    OpInputList arg_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("input_handles", &arg_list));

    // For each input, the allocation it corresponds to and a flag indicating
    // whether or not it should be released, i.e. discarded from the resource
    // manager. One ref on each allocation is owned by this vector, and freed on
    // exit.
    std::vector<XRTTupleAllocation::ExpandedTupleInput> input_vector(
        arg_list.size());
    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    xla::ShapeTree<XRTTupleAllocation::ExpandedTupleInput> tuple_shape_tree;
    // device_ordinal is filled in by ParseTupleTree with the ordinal of one of
    // the allocations. It is guaranteed that there is at least on allocation in
    // any legal tree. We validate below in XRTTupleAllocation::MakeTuple that
    // all the allocations are on the same device.
    int device_ordinal;
    OP_REQUIRES_OK(ctx, XRTStateHelpers::ParseTupleTree(
                            tuple_proto, arg_list, &input_vector,
                            &tuple_shape_tree, &device_ordinal, rm));

    // We are guaranteed that the underlying device object won't be deleted out
    // from under us, while the ScopedRef is live.
    class DeviceAccessor::ScopedRef device_ref;
    OP_REQUIRES_OK(
        ctx, DeviceAccessor::InitScopedRef(ctx, device_ordinal, &device_ref));

    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    XRTTupleAllocation* output_allocation;
    OP_REQUIRES_OK(ctx, XRTTupleAllocation::MakeTuple(
                            memory_manager.get(), device_ref.backend(),
                            device_ref.device_ordinal(), tuple_shape_tree,
                            &output_allocation));
    RefPtr<XRTTupleAllocation> output_ptr(output_allocation);
    for (int i = 0; i < input_vector.size(); ++i) {
      if (input_vector[i].release_allocation_after_use) {
        OP_REQUIRES_OK(ctx,
                       memory_manager->Release(arg_list[i].scalar<int64>()()));
      }
    }

    Tensor output(DT_INT64, TensorShape({}));
    output.scalar<int64>()() = memory_manager->Register(std::move(output_ptr));
    ctx->set_output(0, output);
  }
};

// Op that reads a device-resident tuple to host memory and returns it as a
// literal.
template <bool discard_, class DeviceAccessor>
class XRTReadLiteralOp : public OpKernel {
 public:
  explicit XRTReadLiteralOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~XRTReadLiteralOp() override = default;
  XRTReadLiteralOp(const XRTReadLiteralOp&) = delete;
  XRTReadLiteralOp& operator=(const XRTReadLiteralOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTReadLiteralOp::Compute";
    auto timed = monitoring::MakeTimed(xrt_metrics::GetReadLiteralCell());

    const Tensor& handle_tensor = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(handle_tensor.shape()),
        errors::Internal("computation input should be an int64 scalar"));
    int64 allocation_handle = handle_tensor.scalar<int64>()();

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    RefPtr<XRTTupleAllocation> allocation;
    OP_REQUIRES_OK(ctx, memory_manager->Lookup(allocation_handle, &allocation));

    if (discard_) {
      VLOG(2) << "Releasing handle " << allocation_handle;
      OP_REQUIRES_OK(ctx, memory_manager->Release(allocation_handle));
    }

    // We are guaranteed that the underlying device object won't be deleted out
    // from under us, while the ScopedRef is live.
    class DeviceAccessor::ScopedRef device_ref;
    OP_REQUIRES_OK(ctx, DeviceAccessor::InitScopedRef(
                            ctx, allocation->device_ordinal(), &device_ref));

    xla::Literal literal(allocation->on_host_shape());
    OP_REQUIRES_OK(ctx, allocation->ToLiteral(device_ref.backend(), &literal));
    xla::LiteralProto literal_proto = literal.ToProto();

    Tensor output(DT_STRING, TensorShape({}));
    SerializeToTString(literal_proto, &output.scalar<tstring>()());
    ctx->set_output(0, output);
  }
};

// Op that reads a device-resident tuple to host memory and returns it as a
// literal.
template <class DeviceAccessor>
class XRTReadToTensorOp : public OpKernel {
 public:
  explicit XRTReadToTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("release_handles", &discard_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
  }
  ~XRTReadToTensorOp() override = default;
  XRTReadToTensorOp(const XRTReadToTensorOp&) = delete;
  XRTReadToTensorOp& operator=(const XRTReadToTensorOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTReadToTensorOp::Compute";
    auto timed = monitoring::MakeTimed(xrt_metrics::GetReadToTensorCell());

    const Tensor& handle_tensor = ctx->input(0);
    // TODO(phawkins,dlibenzi): accept multiple handles (i.e., vectors, not
    // just scalars.)
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(handle_tensor.shape()),
        errors::Internal("computation input should be an int64 scalar"));
    int64 allocation_handle = handle_tensor.scalar<int64>()();

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    RefPtr<XRTTupleAllocation> allocation;
    OP_REQUIRES_OK(ctx, memory_manager->Lookup(allocation_handle, &allocation));

    if (discard_) {
      VLOG(2) << "Releasing handle " << allocation_handle;
      OP_REQUIRES_OK(ctx, memory_manager->Release(allocation_handle));
    }

    // We are guaranteed that the underlying device object won't be deleted out
    // from under us, while the ScopedRef is live.
    class DeviceAccessor::ScopedRef device_ref;
    OP_REQUIRES_OK(ctx, DeviceAccessor::InitScopedRef(
                            ctx, allocation->device_ordinal(), &device_ref));

    xla::Shape shape = allocation->on_host_shape();
    int output = 0;
    Status status = xla::ShapeUtil::ForEachMutableSubshapeWithStatus(
        &shape,
        [&](xla::Shape* subshape, const xla::ShapeIndex& index) -> Status {
          if (subshape->IsTuple()) return Status::OK();

          xla::PrimitiveType xla_type;
          TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(
              ctx->expected_output_dtype(output), &xla_type));
          if (xla_type != subshape->element_type()) {
            return errors::InvalidArgument(
                "Type mismatch between buffer type (", subshape->ToString(),
                ") and tensor type (",
                DataTypeString(ctx->expected_output_dtype(output)),
                ") for output tensor ", output);
          }

          TensorShape output_shape;
          TF_RETURN_IF_ERROR(XLAShapeToTensorShape(*subshape, &output_shape));

          Tensor* output_tensor;
          TF_RETURN_IF_ERROR(
              ctx->allocate_output(output, output_shape, &output_tensor));

          XRTTupleAllocation* sub;
          TF_RETURN_IF_ERROR(XRTTupleAllocation::MakeSubBuffer(
              allocation.get(), index, &sub, /*alias_parent_allocation=*/true));
          core::ScopedUnref sub_unref(sub);

          xla::MutableBorrowingLiteral literal;
          TF_RETURN_IF_ERROR(HostTensorToMutableBorrowingLiteral(
              xla::LayoutUtil::GetWithDefaultLayout(*subshape), output_tensor,
              &literal));
          TF_RETURN_IF_ERROR(sub->ToLiteral(device_ref.backend(), &literal));

          ++output;
          return Status::OK();
        });
    OP_REQUIRES_OK(ctx, status);
  }
  bool discard_;
  DataTypeVector dtypes_;
};

// Op that writes a new literal value into device-resident memory.
template <class DeviceAccessor>
class XRTWriteLiteralOp : public OpKernel {
 public:
  explicit XRTWriteLiteralOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~XRTWriteLiteralOp() override = default;
  XRTWriteLiteralOp(const XRTWriteLiteralOp&) = delete;
  XRTWriteLiteralOp& operator=(const XRTWriteLiteralOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTWriteLiteralOp::Compute";
    auto timed = monitoring::MakeTimed(xrt_metrics::GetWriteLiteralCell());

    const Tensor& handle_tensor = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(handle_tensor.shape()),
        errors::Internal("computation input should be an int64 scalar"));
    int64 allocation_handle = handle_tensor.scalar<int64>()();

    const Tensor& literal_info = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(literal_info.shape()),
                errors::Internal("literal input should be a string scalar"));
    xla::LiteralProto literal_proto;
    OP_REQUIRES(
        ctx, ParseFromTString(literal_info.scalar<tstring>()(), &literal_proto),
        errors::InvalidArgument(
            "Unable to parse allocation input to LiteralProto"));
    xla::Literal literal;
    OP_REQUIRES_OK(ctx, XRTStateHelpers::MakeLiteral(literal_proto, &literal));

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    RefPtr<XRTTupleAllocation> allocation;
    OP_REQUIRES_OK(ctx, memory_manager->Lookup(allocation_handle, &allocation));

    // We are guaranteed that the underlying device object won't be deleted out
    // from under us, while the ScopedRef is live.
    typename DeviceAccessor::ScopedRef device_ref;
    OP_REQUIRES_OK(ctx, DeviceAccessor::InitScopedRef(
                            ctx, allocation->device_ordinal(), &device_ref));
    OP_REQUIRES_OK(ctx,
                   allocation->WriteLiteral(device_ref.backend(), literal));

    Tensor output(DT_INT64, TensorShape({}));
    output.scalar<int64>()() = allocation_handle;
    ctx->set_output(0, output);
  }
};

// Op that discards a handle to device memory.
template <class DeviceAccessor>
class XRTReleaseAllocationOp : public OpKernel {
 public:
  explicit XRTReleaseAllocationOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~XRTReleaseAllocationOp() override = default;
  XRTReleaseAllocationOp(const XRTReleaseAllocationOp&) = delete;
  XRTReleaseAllocationOp& operator=(const XRTReleaseAllocationOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTReleaseAllocationOp::Compute";
    auto timed = monitoring::MakeTimed(xrt_metrics::GetReleaseAllocationCell());

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    const Tensor& allocation_handle = ctx->input(0);
    auto flat_keys = allocation_handle.flat<int64>();
    for (int64 i = 0; i < flat_keys.size(); ++i) {
      int64 key = flat_keys(i);
      OP_REQUIRES_OK(ctx, memory_manager->Release(key));
      VLOG(2) << "Released allocation handle " << key;
    }
  }
};

// Op that discards a handle to device memory.
template <class DeviceAccessor>
class XRTReleaseAllAllocationsOp : public OpKernel {
 public:
  explicit XRTReleaseAllAllocationsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  ~XRTReleaseAllAllocationsOp() override = default;
  XRTReleaseAllAllocationsOp(const XRTReleaseAllAllocationsOp&) = delete;
  XRTReleaseAllAllocationsOp& operator=(const XRTReleaseAllAllocationsOp&) =
      delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTReleaseAllAllocationsOp::Compute";
    auto timed =
        monitoring::MakeTimed(xrt_metrics::GetReleaseAllAllocationsCell());

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));
    XRTMemoryManager::Get(rm)->ReleaseAllAllocations();
  }
};

template <class DeviceAccessor>
class XRTCompactAllocationsOp : public OpKernel {
 public:
  explicit XRTCompactAllocationsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~XRTCompactAllocationsOp() override = default;
  XRTCompactAllocationsOp(const XRTCompactAllocationsOp&) = delete;
  XRTCompactAllocationsOp& operator=(const XRTCompactAllocationsOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "XRTCompactAllocationsOp::Compute";
    auto timed =
        monitoring::MakeTimed(xrt_metrics::GetCompactAllocationsCell());

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));
    RefPtr<XRTMemoryManager> memory_manager = XRTMemoryManager::Get(rm);
    class DeviceAccessor::ScopedRef device_ref;
    OP_REQUIRES_OK(ctx, DeviceAccessor::InitScopedRef(ctx, &device_ref));
    OP_REQUIRES_OK(ctx, memory_manager->CompactAllocations(
                            device_ref.backend(), device_ref.device_ordinal()));
  }
};

template <class DeviceAccessor>
class XRTMemoryInfoOp : public OpKernel {
 public:
  explicit XRTMemoryInfoOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~XRTMemoryInfoOp() override = default;
  XRTMemoryInfoOp(const XRTMemoryInfoOp&) = delete;
  XRTMemoryInfoOp& operator=(const XRTMemoryInfoOp&) = delete;

  void Compute(OpKernelContext* ctx) override {
    auto kernel_fn = [&]() -> Status {
      VLOG(1) << "XRTMemoryInfoOp::Compute";

      class DeviceAccessor::ScopedRef device_ref;
      TF_RETURN_IF_ERROR(DeviceAccessor::InitScopedRef(ctx, &device_ref));
      TF_ASSIGN_OR_RETURN(
          se::StreamExecutor * stream_executor,
          device_ref.backend()->stream_executor(device_ref.device_ordinal()));
      int64 mem_free = -1;
      int64 mem_total = -1;
      if (!stream_executor->DeviceMemoryUsage(&mem_free, &mem_total)) {
        VLOG(2) << "Device " << ctx->device()->name()
                << " does not expose memory information";
      }
      xrt::MemoryInfo mem_info;
      mem_info.set_kb_total((mem_total >= 0) ? mem_total / 1024 : -1);
      mem_info.set_kb_free((mem_free >= 0) ? mem_free / 1024 : -1);

      Tensor output(DT_STRING, TensorShape({}));
      output.scalar<tstring>()() = mem_info.SerializeAsString();
      ctx->set_output(0, output);
      return Status::OK();
    };
    OP_REQUIRES_OK(ctx, kernel_fn());
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_KERNELS_XRT_STATE_OPS_H_
