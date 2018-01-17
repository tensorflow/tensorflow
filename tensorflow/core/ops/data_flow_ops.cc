/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status DequeueManyV2Shape(InferenceContext* c, ShapeHandle n_shape) {
  auto* t = c->input_handle_shapes_and_types(0);
  if (t != nullptr && t->size() == c->num_outputs()) {
    for (int i = 0; i < c->num_outputs(); ++i) {
      ShapeHandle combined_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(n_shape, (*t)[i].shape, &combined_shape));
      c->set_output(i, combined_shape);
    }
    return Status::OK();
  } else {
    return shape_inference::UnknownShape(c);
  }
}

}  // namespace

// --------------------------------------------------------------------------

REGISTER_OP("DynamicPartition")
    .Input("data: T")
    .Input("partitions: int32")
    .Output("outputs: num_partitions * T")
    .Attr("num_partitions: int")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      int64 num_partitions;
      TF_RETURN_IF_ERROR(c->GetAttr("num_partitions", &num_partitions));

      ShapeHandle data_shape = c->input(0);
      ShapeHandle partitions_shape = c->input(1);

      if (!c->RankKnown(partitions_shape)) {
        return shape_inference::UnknownShape(c);
      }

      const int64 rank = c->Rank(partitions_shape);

      // data shape must start with partitions_shape
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->MergePrefix(data_shape, partitions_shape, &unused, &unused));

      // The partition shape is dynamic in the 0th dimension, and matches
      // data_shape in the remaining dimensions.
      ShapeHandle unknown_dim0 = c->MakeShape({c->UnknownDim()});

      ShapeHandle data_suffix_shape;
      TF_RETURN_IF_ERROR(c->Subshape(data_shape, rank, &data_suffix_shape));
      ShapeHandle result_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(unknown_dim0, data_suffix_shape, &result_shape));

      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, result_shape);
      }

      return Status::OK();
    });

namespace {

Status DynamicStitchShapeFunction(InferenceContext* c) {
  int32 num_partitions;
  TF_RETURN_IF_ERROR(c->GetAttr("N", &num_partitions));

  bool all_indices_constant = true;
  int32 max_index = 0;
  ShapeHandle extra_shape = c->UnknownShape();
  for (int i = 0; i < num_partitions; ++i) {
    const Tensor* indices_t = c->input_tensor(i);
    if (indices_t == nullptr) {
      all_indices_constant = false;
    }

    ShapeHandle indices_shape = c->input(i);
    ShapeHandle data_shape = c->input(i + num_partitions);
    if (!c->RankKnown(indices_shape)) {
      continue;
    }
    const int64 indices_rank = c->Rank(indices_shape);

    // Assert that data_shape starts with indices_shape.
    ShapeHandle unused;
    TF_RETURN_IF_ERROR(
        c->MergePrefix(data_shape, indices_shape, &unused, &unused));

    // The rest belongs to output.
    ShapeHandle rest;
    TF_RETURN_IF_ERROR(c->Subshape(data_shape, indices_rank, &rest));
    TF_RETURN_IF_ERROR(c->Merge(extra_shape, rest, &extra_shape));

    if (indices_t != nullptr) {
      // The length is based on the highest index from flattened indices.
      const int32* indices = indices_t->flat<int32>().data();
      int64 count = indices_t->NumElements();
      for (int64 i = 0; i < count; ++i) {
        if (indices[i] > max_index) {
          max_index = indices[i];
        }
      }
    }
  }

  ShapeHandle output_shape = c->Vector(
      all_indices_constant ? c->MakeDim(max_index + 1) : c->UnknownDim());
  TF_RETURN_IF_ERROR(c->Concatenate(output_shape, extra_shape, &output_shape));
  c->set_output(0, output_shape);
  return Status::OK();
}

}  // namespace

REGISTER_OP("DynamicStitch")
    .Input("indices: N * int32")
    .Input("data: N * T")
    .Output("merged: T")
    .Attr("N : int >= 1")
    .Attr("T : type")
    .SetShapeFn(DynamicStitchShapeFunction);

REGISTER_OP("ParallelDynamicStitch")
    .Input("indices: N * int32")
    .Input("data: N * T")
    .Output("merged: T")
    .Attr("N : int >= 1")
    .Attr("T : type")
    .SetShapeFn(DynamicStitchShapeFunction);

// --------------------------------------------------------------------------

namespace {
Status TwoElementVectorInputsAndScalarOutputs(InferenceContext* c) {
  ShapeHandle handle;
  DimensionHandle unused_handle;
  for (int i = 0; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status ScalarAndTwoElementVectorInputsAndScalarOutputs(InferenceContext* c) {
  ShapeHandle handle;
  DimensionHandle unused_handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  for (int i = 1; i < c->num_inputs(); ++i) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &handle));
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_handle));
  }
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

Status TwoElementOutput(InferenceContext* c) {
  c->set_output(0, c->Vector(2));
  return Status::OK();
}

Status ScalarOutput(InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return Status::OK();
}
}  // namespace

REGISTER_OP("RandomShuffleQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("min_after_dequeue: int = 0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("RandomShuffleQueueV2")
    .Output("handle: resource")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("min_after_dequeue: int = 0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FIFOQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("FIFOQueueV2")
    .Output("handle: resource")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("PaddingFIFOQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("PaddingFIFOQueueV2")
    .Output("handle: resource")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("PriorityQueue")
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 0 = []")
    .Attr("shapes: list(shape) >= 0")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("PriorityQueueV2")
    .Output("handle: resource")
    .Attr("component_types: list(type) >= 0 = []")
    .Attr("shapes: list(shape) >= 0")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FakeQueue")
    .Input("resource: resource")
    .Output("handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("QueueEnqueue")
    .Input("handle: Ref(string)")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueEnqueueV2")
    .Input("handle: resource")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueEnqueueMany")
    .Input("handle: Ref(string)")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueEnqueueManyV2")
    .Input("handle: resource")
    .Input("components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueDequeue")
    .Input("handle: Ref(string)")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueDequeueV2")
    .Input("handle: resource")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn([](InferenceContext* c) {
      auto* t = c->input_handle_shapes_and_types(0);
      if (t != nullptr && t->size() == c->num_outputs()) {
        for (int i = 0; i < c->num_outputs(); ++i) {
          c->set_output(i, (*t)[i].shape);
        }
        return Status::OK();
      } else {
        return shape_inference::UnknownShape(c);
      }
    });

REGISTER_OP("QueueDequeueMany")
    .Input("handle: Ref(string)")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueDequeueManyV2")
    .Input("handle: resource")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle n_shape;
      if (c->input_tensor(1) == nullptr) {
        n_shape = c->Vector(InferenceContext::kUnknownDim);
      } else {
        const int32 n = c->input_tensor(1)->scalar<int32>()();
        if (n < 0) {
          return errors::InvalidArgument("Input 'n' must be >= 0, but is ", n);
        }
        n_shape = c->Vector(n);
      }
      return DequeueManyV2Shape(c, n_shape);
    });

REGISTER_OP("QueueDequeueUpTo")
    .Input("handle: Ref(string)")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("QueueDequeueUpToV2")
    .Input("handle: resource")
    .Input("n: int32")
    .Output("components: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn([](InferenceContext* c) {
      return DequeueManyV2Shape(c, c->Vector(InferenceContext::kUnknownDim));
    });

REGISTER_OP("QueueClose")
    .Input("handle: Ref(string)")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Attr("cancel_pending_enqueues: bool = false");

REGISTER_OP("QueueCloseV2")
    .Input("handle: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .Attr("cancel_pending_enqueues: bool = false");

REGISTER_OP("QueueIsClosed")
    .Input("handle: Ref(string)")
    .Output("is_closed: bool")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("QueueIsClosedV2")
    .Input("handle: resource")
    .Output("is_closed: bool")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("QueueSize")
    .Input("handle: Ref(string)")
    .Output("size: int32")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

REGISTER_OP("QueueSizeV2")
    .Input("handle: resource")
    .Output("size: int32")
    .SetShapeFn(shape_inference::UnchangedShape);

// --------------------------------------------------------------------------

REGISTER_OP("AccumulatorNumAccumulated")
    .Input("handle: Ref(string)")
    .Output("num_accumulated: int32")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("AccumulatorSetGlobalStep")
    .Input("handle: Ref(string)")
    .Input("new_global_step: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("ConditionalAccumulator")
    .Output("handle: Ref(string)")
    .Attr("dtype: numbertype")
    .Attr("shape: shape")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("AccumulatorApplyGradient")
    .Input("handle: Ref(string)")
    .Input("local_step: int64")
    .Input("gradient: dtype")
    .Attr("dtype: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("AccumulatorTakeGradient")
    .Input("handle: Ref(string)")
    .Input("num_required: int32")
    .Output("average: dtype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // Shape of output is the shape of the accumulator referenced
      // by 'handle', but which is not available here, so we lose
      // shape information.
      return shape_inference::UnknownShape(c);
    })
    .Attr("dtype: numbertype");

REGISTER_OP("SparseConditionalAccumulator")
    .Output("handle: Ref(string)")
    .Attr("dtype: numbertype")
    .Attr("shape: shape")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    });

REGISTER_OP("SparseAccumulatorApplyGradient")
    .Input("handle: Ref(string)")
    .Input("local_step: int64")
    .Input("gradient_indices: int64")
    .Input("gradient_values: dtype")
    .Input("gradient_shape: int64")
    .Attr("dtype: numbertype")
    .Attr("has_known_shape: bool")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("SparseAccumulatorTakeGradient")
    .Input("handle: Ref(string)")
    .Input("num_required: int32")
    .Output("indices: int64")
    .Output("values: dtype")
    .Output("shape: int64")
    .Attr("dtype: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // Shape of output is the shape of the accumulator referenced
      // by 'handle', but which is not available here, so we lose
      // shape information.
      return shape_inference::UnknownShape(c);
    });

// --------------------------------------------------------------------------

REGISTER_OP("StackV2")
    .Input("max_size: int32")
    .Output("handle: resource")
    .Attr("elem_type: type")
    .Attr("stack_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("StackPushV2")
    .Input("handle: resource")
    .Input("elem: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("swap_memory: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("StackPopV2")
    .Input("handle: resource")
    .Output("elem: elem_type")
    .Attr("elem_type: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("StackCloseV2")
    .Input("handle: resource")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

// Deprecated ref-typed variants of stack.

REGISTER_OP("Stack")
    .Output("handle: Ref(string)")
    .Attr("elem_type: type")
    .Attr("stack_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("StackPush")
    .Input("handle: Ref(string)")
    .Input("elem: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("swap_memory: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("StackPop")
    .Input("handle: Ref(string)")
    .Output("elem: elem_type")
    .Attr("elem_type: type")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("StackClose")
    .Input("handle: Ref(string)")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

// --------------------------------------------------------------------------

REGISTER_OP("TensorArrayV3")
    .Input("size: int32")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .Attr("dynamic_size: bool = false")
    .Attr("clear_after_read: bool = true")
    .Attr("identical_element_shapes: bool = false")
    .Attr("tensor_array_name: string = ''")
    .Output("handle: resource")
    .Output("flow: float")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->Vector(2));
      c->set_output(1, c->Scalar());
      bool identical_shapes;
      TF_RETURN_IF_ERROR(
          c->GetAttr("identical_element_shapes", &identical_shapes));
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("element_shape", &p));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      if (c->FullyDefined(s) || identical_shapes) {
        c->set_output_handle_shapes_and_types(
            0, std::vector<shape_inference::ShapeAndType>{{s, t}});
      }
      return Status::OK();
    });

REGISTER_OP("TensorArrayGradV3")
    .Input("handle: resource")
    .Input("flow_in: float")
    .Output("grad_handle: resource")
    .Output("flow_out: float")
    .Attr("source: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      c->set_output(0, c->Vector(2));
      c->set_output(1, c->Scalar());
      if (c->input_handle_shapes_and_types(0)) {
        c->set_output_handle_shapes_and_types(
            0, *c->input_handle_shapes_and_types(0));
      }
      return Status::OK();
    });

REGISTER_OP("TensorArrayWriteV3")
    .Input("handle: resource")
    .Input("index: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));

      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && !handle_data->empty()) {
        shape_inference::ShapeAndType shape_and_type = (*handle_data)[0];
        ShapeHandle value_shape = c->input(2);
        TF_RETURN_IF_ERROR(
            c->Merge(shape_and_type.shape, value_shape, &unused));
      }

      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TensorArrayReadV3")
    .Input("handle: resource")
    .Input("index: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      auto shapes = c->input_handle_shapes_and_types(0);
      if (shapes != nullptr && !shapes->empty()) {
        ShapeHandle tensor_shape = shapes->at(0).shape;
        c->set_output(0, tensor_shape);
        return Status::OK();
      } else {
        return shape_inference::UnknownShape(c);
      }
    });

REGISTER_OP("TensorArrayGatherV3")
    .Input("handle: resource")
    .Input("indices: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::UnknownShape(c);
    });

REGISTER_OP("TensorArrayScatterV3")
    .Input("handle: resource")
    .Input("indices: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TensorArrayConcatV3")
    .Input("handle: resource")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Output("lengths: int64")
    .Attr("dtype: type")
    .Attr("element_shape_except0: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->UnknownShape());
      c->set_output(1, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_OP("TensorArraySplitV3")
    .Input("handle: resource")
    .Input("value: T")
    .Input("lengths: int64")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TensorArraySizeV3")
    .Input("handle: resource")
    .Input("flow_in: float")
    .Output("size: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("TensorArrayCloseV3")
    .Input("handle: resource")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return Status::OK();
    });

// --------------------------------------------------------------------------

// Deprecated TensorArray methods

REGISTER_OP("TensorArray")
    .Input("size: int32")
    .Attr("dtype: type")
    .Attr("dynamic_size: bool = false")
    .Attr("clear_after_read: bool = true")
    .Attr("tensor_array_name: string = ''")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .Output("handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayV3");
// TODO(cwhipkey): mark this deprecated in favor of V3.
REGISTER_OP("TensorArrayV2")
    .Input("size: int32")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .Attr("dynamic_size: bool = false")
    .Attr("clear_after_read: bool = true")
    .Attr("tensor_array_name: string = ''")
    .Output("handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      c->set_output(0, c->Vector(2));
      return Status::OK();
    });
REGISTER_OP("TensorArrayGrad")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("grad_handle: Ref(string)")
    .Attr("source: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayGradV3");
// TODO(cwhipkey): mark this deprecated in favor of V3.
REGISTER_OP("TensorArrayGradV2")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("grad_handle: string")
    .Attr("source: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      c->set_output(0, c->Vector(2));
      return Status::OK();
    });
REGISTER_OP("TensorArrayWrite")
    .Input("handle: Ref(string)")
    .Input("index: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayWriteV3");
// TODO(cwhipkey): mark this deprecated in favor of V3.
REGISTER_OP("TensorArrayWriteV2")
    .Input("handle: string")
    .Input("index: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });
REGISTER_OP("TensorArrayRead")
    .Input("handle: Ref(string)")
    .Input("index: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayReadV3");
// TODO(cwhipkey): mark this deprecated in favor of V3.
REGISTER_OP("TensorArrayReadV2")
    .Input("handle: string")
    .Input("index: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::UnknownShape(c);
    });
REGISTER_OP("TensorArrayPack")
    .Input("handle: Ref(string)")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayGatherV3 with RangeOp");
REGISTER_OP("TensorArrayUnpack")
    .Input("handle: Ref(string)")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(20, "Use TensorArrayScatterV3 with RangeOp");
REGISTER_OP("TensorArrayGather")
    .Input("handle: Ref(string)")
    .Input("indices: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayGatherV3");
// TODO(cwhipkey): mark this deprecated in favor of V3.
REGISTER_OP("TensorArrayGatherV2")
    .Input("handle: string")
    .Input("indices: int32")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Attr("dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::UnknownShape(c);
    });
REGISTER_OP("TensorArrayScatter")
    .Input("handle: Ref(string)")
    .Input("indices: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(19, "Use TensorArrayGradV3");
// TODO(cwhipkey): mark this deprecated in favor of V3.
REGISTER_OP("TensorArrayScatterV2")
    .Input("handle: string")
    .Input("indices: int32")
    .Input("value: T")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(0), 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });
REGISTER_OP("TensorArrayConcat")
    .Input("handle: Ref(string)")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Output("lengths: int64")
    .Attr("dtype: type")
    .Attr("element_shape_except0: shape = { unknown_rank: true }")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArrayGradV3");
REGISTER_OP("TensorArrayConcatV2")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("value: dtype")
    .Output("lengths: int64")
    .Attr("dtype: type")
    .Attr("element_shape_except0: shape = { unknown_rank: true }")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      c->set_output(0, c->UnknownShape());
      c->set_output(1, c->Vector(c->UnknownDim()));
      return Status::OK();
    });
REGISTER_OP("TensorArraySplit")
    .Input("handle: Ref(string)")
    .Input("value: T")
    .Input("lengths: int64")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArraySplitV3");
// TODO(cwhipkey): mark this deprecated in favor of V3.
REGISTER_OP("TensorArraySplitV2")
    .Input("handle: string")
    .Input("value: T")
    .Input("lengths: int64")
    .Input("flow_in: float")
    .Output("flow_out: float")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });
REGISTER_OP("TensorArraySize")
    .Input("handle: Ref(string)")
    .Input("flow_in: float")
    .Output("size: int32")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(16, "Use TensorArraySizeV3");
// TODO(cwhipkey): mark this deprecated in favor of V3.
REGISTER_OP("TensorArraySizeV2")
    .Input("handle: string")
    .Input("flow_in: float")
    .Output("size: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return shape_inference::ScalarShape(c);
    });
REGISTER_OP("TensorArrayClose")
    .Input("handle: Ref(string)")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); })
    .Deprecated(16, "Use TensorArrayCloseV3");
// TODO(cwhipkey): mark this deprecated in favor of V3.
REGISTER_OP("TensorArrayCloseV2")
    .Input("handle: string")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      return Status::OK();
    });

// --------------------------------------------------------------------------

REGISTER_OP("Barrier")
    .SetIsStateful()
    .Output("handle: Ref(string)")
    .Attr("component_types: list(type) >= 1")
    .Attr("shapes: list(shape) >= 0 = []")
    .Attr("capacity: int = -1")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(TwoElementOutput);

REGISTER_OP("BarrierInsertMany")
    .Input("handle: Ref(string)")
    .Input("keys: string")
    .Input("values: T")
    .Attr("T: type")
    .Attr("component_index: int")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle keys = c->input(1);
      ShapeHandle values = c->input(2);
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &handle));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(handle, 0), 2, &unused_dim));
      TF_RETURN_IF_ERROR(c->WithRank(keys, 1, &keys));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 1, &values));
      TF_RETURN_IF_ERROR(c->Merge(keys, c->Vector(c->Dim(values, 0)), &handle));
      return Status::OK();
    });

REGISTER_OP("BarrierTakeMany")
    .Input("handle: Ref(string)")
    .Input("num_elements: int32")
    .Output("indices: int64")
    .Output("keys: string")
    .Output("values: component_types")
    .Attr("component_types: list(type) >= 1")
    .Attr("allow_small_batch: bool = false")
    .Attr("wait_for_incomplete: bool = false")
    .Attr("timeout_ms: int = -1")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BarrierClose")
    .Input("handle: Ref(string)")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs)
    .Attr("cancel_pending_enqueues: bool = false");

REGISTER_OP("BarrierReadySize")
    .Input("handle: Ref(string)")
    .Output("size: int32")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

REGISTER_OP("BarrierIncompleteSize")
    .Input("handle: Ref(string)")
    .Output("size: int32")
    .SetShapeFn(TwoElementVectorInputsAndScalarOutputs);

// --------------------------------------------------------------------------

REGISTER_OP("GetSessionHandle")
    .Input("value: T")
    .Output("handle: string")
    .Attr("T: type")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GetSessionHandleV2")
    .Input("value: T")
    .Output("handle: resource")
    .Attr("T: type")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GetSessionTensor")
    .Input("handle: string")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      return shape_inference::UnknownShape(c);
    });

REGISTER_OP("DeleteSessionTensor")
    .Input("handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      return Status::OK();
    });

REGISTER_OP("Stage")
    .Input("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("Unstage")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("StagePeek")
    .Input("index: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("StageSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("StageClear")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

// UnorderedMap
REGISTER_OP("MapStage")
    .Input("key: int64")
    .Input("indices: int32")
    .Input("values: fake_dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("fake_dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .SetIsStateful();

REGISTER_OP("MapPeek")
    .Input("key: int64")
    .Input("indices: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("MapUnstage")
    .Input("key: int64")
    .Input("indices: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("MapUnstageNoKey")
    .Input("indices: int32")
    .Output("key: int64")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("MapSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("MapIncompleteSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("MapClear")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .SetIsStateful();

// OrderedMap
REGISTER_OP("OrderedMapStage")
    .Input("key: int64")
    .Input("indices: int32")
    .Input("values: fake_dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("fake_dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .SetIsStateful();

REGISTER_OP("OrderedMapPeek")
    .Input("key: int64")
    .Input("indices: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapUnstage")
    .Input("key: int64")
    .Input("indices: int32")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapUnstageNoKey")
    .Input("indices: int32")
    .Output("key: int64")
    .Output("values: dtypes")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapIncompleteSize")
    .Output("size: int32")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .SetIsStateful();

REGISTER_OP("OrderedMapClear")
    .Attr("capacity: int >= 0 = 0")
    .Attr("memory_limit: int >= 0 = 0")
    .Attr("dtypes: list(type)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .SetIsStateful();

REGISTER_OP("RecordInput")
    .Output("records: string")
    .Attr("file_pattern: string")
    .Attr("file_random_seed: int = 301")
    .Attr("file_shuffle_shift_ratio: float = 0")
    .Attr("file_buffer_size: int = 10000")
    .Attr("file_parallelism: int = 16")
    .Attr("batch_size: int = 32")
    .Attr("compression_type: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
