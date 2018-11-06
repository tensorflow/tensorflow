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

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_device.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
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
        TF_RETURN_IF_ERROR(
            XRTTupleAllocation::Lookup(rm, key, &input.allocation));
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

    const Tensor& allocation_info = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(allocation_info.shape()),
                errors::Internal("allocation input should be a string scalar"));
    xrt::XLAAllocation allocation_proto;
    OP_REQUIRES(
        ctx,
        allocation_proto.ParseFromString(allocation_info.scalar<string>()()),
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
    OP_REQUIRES_OK(ctx,
                   DeviceAccessor::InitScopedRef(
                       ctx, allocation_proto.device_ordinal(), &device_ref));

    XRTTupleAllocation* allocation;
    OP_REQUIRES_OK(ctx, XRTTupleAllocation::CreateAndTransfer(
                            literal, device_ref.backend(),
                            device_ref.device_ordinal(), &allocation));

    // Intern takes ownership of our reference to allocation.
    int64 key;
    OP_REQUIRES_OK(ctx, allocation->Intern(rm, &key));

    Tensor output(DT_INT64, TensorShape({}));
    output.scalar<int64>()() = key;
    ctx->set_output(0, output);
  }
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

    XRTTupleAllocation* allocation;
    OP_REQUIRES_OK(
        ctx, XRTTupleAllocation::Lookup(rm, allocation_handle, &allocation));
    core::ScopedUnref allocation_unref(allocation);

    if (discard_) {
      VLOG(2) << "Releasing handle " << allocation_handle;
      OP_REQUIRES_OK(ctx, XRTTupleAllocation::DeleteFromResourceManager(
                              rm, allocation_handle));
    }

    XRTTupleAllocation* suballocation;
    OP_REQUIRES_OK(
        ctx, XRTTupleAllocation::MakeSubBuffer(allocation, shape_index,
                                               &suballocation, !discard_));

    // Intern takes ownership of our reference to suballocation.
    int64 key;
    OP_REQUIRES_OK(ctx, suballocation->Intern(rm, &key));

    Tensor output(DT_INT64, TensorShape({}));
    output.scalar<int64>()() = key;
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

    const Tensor& tuple_info = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(tuple_info.shape()),
        errors::Internal("tuple description input should be a string scalar"));
    xrt::XLATupleNode tuple_proto;
    OP_REQUIRES(
        ctx, tuple_proto.ParseFromString(tuple_info.scalar<string>()()),
        errors::InvalidArgument("Unable to parse tuple input to XLATupleNode"));

    OpInputList arg_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("input_handles", &arg_list));

    // For each input, the allocation it corresponds to and a flag indicating
    // whether or not it should be released, i.e. discarded from the resource
    // manager. One ref on each allocation is owned by this vector, and freed on
    // exit.
    std::vector<XRTTupleAllocation::ExpandedTupleInput> input_vector(
        arg_list.size());
    auto cleanup = gtl::MakeCleanup([&input_vector] {
      for (auto& input : input_vector) {
        if (input.allocation != nullptr) {
          input.allocation->Unref();
        }
      }
    });

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

    XRTTupleAllocation* output_allocation;
    OP_REQUIRES_OK(ctx, XRTTupleAllocation::MakeTuple(
                            device_ref.backend(), device_ref.device_ordinal(),
                            tuple_shape_tree, &output_allocation));
    // Add a ScopedUnref to simplify the error path while calling
    // DeleteFromResourceManager.
    core::ScopedUnref unref(output_allocation);
    for (int i = 0; i < input_vector.size(); ++i) {
      if (input_vector[i].release_allocation_after_use) {
        OP_REQUIRES_OK(ctx, XRTTupleAllocation::DeleteFromResourceManager(
                                rm, arg_list[i].scalar<int64>()()));
      }
    }

    // Intern takes ownership of a reference to output_allocation, so add
    // another since the ScopedUnref will release one when this method exits.
    output_allocation->Ref();
    int64 key;
    OP_REQUIRES_OK(ctx, output_allocation->Intern(rm, &key));

    Tensor output(DT_INT64, TensorShape({}));
    output.scalar<int64>()() = key;
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

    const Tensor& handle_tensor = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(handle_tensor.shape()),
        errors::Internal("computation input should be an int64 scalar"));
    int64 allocation_handle = handle_tensor.scalar<int64>()();

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    XRTTupleAllocation* allocation;
    OP_REQUIRES_OK(
        ctx, XRTTupleAllocation::Lookup(rm, allocation_handle, &allocation));
    core::ScopedUnref allocation_unref(allocation);

    if (discard_) {
      VLOG(2) << "Releasing handle " << allocation_handle;
      OP_REQUIRES_OK(ctx, XRTTupleAllocation::DeleteFromResourceManager(
                              rm, allocation_handle));
    }

    // We are guaranteed that the underlying device object won't be deleted out
    // from under us, while the ScopedRef is live.
    class DeviceAccessor::ScopedRef device_ref;
    OP_REQUIRES_OK(ctx, DeviceAccessor::InitScopedRef(
                            ctx, allocation->device_ordinal(), &device_ref));

    xla::Literal literal;
    OP_REQUIRES_OK(
        ctx, allocation->ToLiteral(device_ref.backend(),
                                   device_ref.device_ordinal(), &literal));
    xla::LiteralProto literal_proto = literal.ToProto();

    Tensor output(DT_STRING, TensorShape({}));
    literal_proto.SerializeToString(&output.scalar<string>()());
    ctx->set_output(0, output);
  }
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

    const Tensor& handle_tensor = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(handle_tensor.shape()),
        errors::Internal("computation input should be an int64 scalar"));
    int64 allocation_handle = handle_tensor.scalar<int64>()();

    const Tensor& literal_info = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(literal_info.shape()),
                errors::Internal("literal input should be a string scalar"));
    xla::LiteralProto literal_proto;
    OP_REQUIRES(ctx,
                literal_proto.ParseFromString(literal_info.scalar<string>()()),
                errors::InvalidArgument(
                    "Unable to parse allocation input to LiteralProto"));
    xla::Literal literal;
    OP_REQUIRES_OK(ctx, XRTStateHelpers::MakeLiteral(literal_proto, &literal));

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    XRTTupleAllocation* allocation;
    OP_REQUIRES_OK(
        ctx, XRTTupleAllocation::Lookup(rm, allocation_handle, &allocation));
    core::ScopedUnref allocation_unref(allocation);
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

    const Tensor& allocation_handle = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(allocation_handle.shape()),
                errors::Internal("handle input should be an int64 scalar"));
    int64 key = allocation_handle.scalar<int64>()();

    ResourceMgr* rm;
    OP_REQUIRES_OK(ctx, DeviceAccessor::GetResourceManager(ctx, &rm));

    OP_REQUIRES_OK(ctx, XRTTupleAllocation::DeleteFromResourceManager(rm, key));

    VLOG(2) << "Released allocation handle " << key;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_KERNELS_XRT_STATE_OPS_H_
