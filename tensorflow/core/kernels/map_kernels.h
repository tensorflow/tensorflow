/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_MAP_KERNELS_H_
#define TENSORFLOW_CORE_KERNELS_MAP_KERNELS_H_

#include <iostream>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/tensor_map.h"

namespace tensorflow {

Status GetInputMap(OpKernelContext* c, int index, const TensorMap** map) {
  if (!TensorShapeUtils::IsScalar(c->input(index).shape())) {
    return errors::InvalidArgument("Input map must be a scalar. Saw: ",
                                   c->input(index).shape().DebugString());
  }
  const TensorMap* m = c->input(index).scalar<Variant>()().get<TensorMap>();
  if (m == nullptr) {
    return errors::InvalidArgument(
        "Input handle is not a map. Saw: '",
        c->input(index).scalar<Variant>()().DebugString(), "'");
  }
  *map = m;
  return Status::OK();
}

// TODO(kattian): change into templated function
Status ForwardInputOrCreateNewMap(OpKernelContext* c, int32 input_index,
                                  int32 output_index,
                                  const TensorMap& input_map,
                                  TensorMap** output_map) {
  // Attempt to forward the input tensor to the output if possible.
  std::unique_ptr<Tensor> maybe_output = c->forward_input(
      input_index, output_index, DT_VARIANT, TensorShape{},
      c->input_memory_type(input_index), AllocatorAttributes());
  Tensor* output_tensor;
  if (maybe_output != nullptr && maybe_output->dtype() == DT_VARIANT &&
      maybe_output->NumElements() == 1) {
    output_tensor = maybe_output.get();
    TensorMap* tmp_out = output_tensor->scalar<Variant>()().get<TensorMap>();
    if (tmp_out == nullptr) {
      return errors::InvalidArgument(
          "Expected input ", input_index, " to be a TensorMap but saw ",
          output_tensor->scalar<Variant>()().TypeName());
    }
    if (tmp_out->RefCountIsOne()) {
      // Woohoo, forwarding succeeded!
      c->set_output(output_index, *output_tensor);
      *output_map = tmp_out;
      return Status::OK();
    }
  }

  // If forwarding is not possible allocate a new output tensor and copy
  // the `input_map` to it.
  AllocatorAttributes attr;
  attr.set_on_host(true);
  TF_RETURN_IF_ERROR(
      c->allocate_output(output_index, {}, &output_tensor, attr));
  output_tensor->scalar<Variant>()() = input_map.Copy();

  *output_map = output_tensor->scalar<Variant>()().get<TensorMap>();
  return Status::OK();
}

class EmptyTensorMap : public OpKernel {
 public:
  explicit EmptyTensorMap(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result, attr));
    TensorMap empty;
    result->scalar<Variant>()() = std::move(empty);
  }
};

class TensorMapSize : public OpKernel {
 public:
  explicit TensorMapSize(OpKernelConstruction* c) : OpKernel(c) {}
  ~TensorMapSize() override {}

  void Compute(OpKernelContext* c) override {
    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));
    Tensor* result;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result));
    result->scalar<int32>()() = m->tensors().size();
  }
};

class TensorMapInsert : public OpKernel {
 public:
  explicit TensorMapInsert(OpKernelConstruction* c) : OpKernel(c) {}
  ~TensorMapInsert() override {}

  void Compute(OpKernelContext* c) override {
    const TensorKey& key = c->input(1);
    const Tensor& value = c->input(2);
    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));

    TensorMap* output_map = nullptr;
    OP_REQUIRES_OK(c, ForwardInputOrCreateNewMap(c, 0, 0, *m, &output_map));
    output_map->replace(key, value);
  }
};

class TensorMapLookup : public OpKernel {
 public:
  explicit TensorMapLookup(OpKernelConstruction* c) : OpKernel(c) {}
  ~TensorMapLookup() override {}

  void Compute(OpKernelContext* c) override {
    const TensorKey& key = c->input(1);
    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));

    OP_REQUIRES(c, m->tensors().find(key) != m->tensors().end(),
                errors::InvalidArgument("Trying to lookup non-existent key."));

    c->set_output(0, m->tensors().find(key)->second);
  }
};

class TensorMapErase : public OpKernel {
 public:
  explicit TensorMapErase(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));
    const TensorKey& key = c->input(1);

    OP_REQUIRES(c, m->tensors().find(key) != m->tensors().end(),
                errors::InvalidArgument("Trying to erase non-existent item."));

    const Tensor& t = m->tensors().find(key)->second;
    c->set_output(1, t);

    TensorMap* output_map = nullptr;
    OP_REQUIRES_OK(c, ForwardInputOrCreateNewMap(c, 0, 0, *m, &output_map));
    output_map->tensors().erase(key);
  }
};

class TensorMapHasKey : public OpKernel {
 public:
  explicit TensorMapHasKey(OpKernelConstruction* c) : OpKernel(c) {}
  ~TensorMapHasKey() override {}

  void Compute(OpKernelContext* c) override {
    const TensorKey& key = c->input(1);
    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));
    Tensor* result;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result));
    result->scalar<bool>()() = m->tensors().find(key) != m->tensors().end();
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAP_KERNELS_H_
