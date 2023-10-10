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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/tensor_map.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/tensor_ops_util.h"

namespace tensorflow {

inline Status GetInputMap(OpKernelContext* ctx, int index,
                          const TensorMap** ret_map) {
  if (!TensorShapeUtils::IsScalar(ctx->input(index).shape())) {
    return errors::InvalidArgument("Input map must be a scalar. Saw: ",
                                   ctx->input(index).shape().DebugString());
  }
  const TensorMap* map = ctx->input(index).scalar<Variant>()().get<TensorMap>();
  if (map == nullptr) {
    return errors::InvalidArgument(
        "Input handle is not a map. Saw: '",
        ctx->input(index).scalar<Variant>()().DebugString(), "'");
  }
  *ret_map = map;
  return OkStatus();
}

// TODO(kattian): change into templated function
inline Status ForwardInputOrCreateNewMap(OpKernelContext* ctx,
                                         int32_t input_index,
                                         int32_t output_index,
                                         const TensorMap& input_map,
                                         TensorMap** output_map) {
  // Attempt to forward the input tensor to the output if possible.
  std::unique_ptr<Tensor> maybe_output = ctx->forward_input(
      input_index, output_index, DT_VARIANT, TensorShape{},
      ctx->input_memory_type(input_index), AllocatorAttributes());
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
      ctx->set_output(output_index, *output_tensor);
      *output_map = tmp_out;
      return OkStatus();
    }
  }

  // If forwarding is not possible allocate a new output tensor and copy
  // the `input_map` to it.
  AllocatorAttributes attr;
  attr.set_on_host(true);
  TF_RETURN_IF_ERROR(
      ctx->allocate_output(output_index, {}, &output_tensor, attr));
  output_tensor->scalar<Variant>()() = input_map.Copy();

  *output_map = output_tensor->scalar<Variant>()().get<TensorMap>();
  return OkStatus();
}

class EmptyTensorMap : public OpKernel {
 public:
  explicit EmptyTensorMap(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result, attr));
    TensorMap empty;
    result->scalar<Variant>()() = std::move(empty);
  }
};

class TensorMapSize : public OpKernel {
 public:
  explicit TensorMapSize(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~TensorMapSize() override {}

  void Compute(OpKernelContext* ctx) override {
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result));
    result->scalar<int32>()() = map->tensors().size();
  }
};

class TensorMapLookup : public OpKernel {
 public:
  explicit TensorMapLookup(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~TensorMapLookup() override {}

  void Compute(OpKernelContext* ctx) override {
    const TensorKey& key = ctx->input(1);
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));

    OP_REQUIRES(
        ctx, map->tensors().find(key) != map->tensors().end(),
        errors::InvalidArgument("Trying to lookup non-existent key. Could not "
                                "find key \"" +
                                key.SummarizeValue(100) + "\"."));

    ctx->set_output(0, map->tensors().find(key)->second);
  }
};

class TensorMapInsert : public OpKernel {
 public:
  explicit TensorMapInsert(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~TensorMapInsert() override {}

  void Compute(OpKernelContext* ctx) override {
    const TensorKey& key = ctx->input(1);
    const Tensor& value = ctx->input(2);
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));

    TensorMap* output_map = nullptr;
    OP_REQUIRES_OK(ctx,
                   ForwardInputOrCreateNewMap(ctx, 0, 0, *map, &output_map));
    output_map->replace(key, value);
  }
};

class TensorMapErase : public OpKernel {
 public:
  explicit TensorMapErase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const TensorKey& key = ctx->input(1);
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));

    OP_REQUIRES(
        ctx, map->tensors().find(key) != map->tensors().end(),
        errors::InvalidArgument("Trying to erase non-existent item. Could not "
                                "find key \"" +
                                key.SummarizeValue(100) + "\"."));

    TensorMap* output_map = nullptr;
    OP_REQUIRES_OK(ctx,
                   ForwardInputOrCreateNewMap(ctx, 0, 0, *map, &output_map));
    output_map->tensors().erase(key);
  }
};

class TensorMapHasKey : public OpKernel {
 public:
  explicit TensorMapHasKey(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  ~TensorMapHasKey() override {}

  void Compute(OpKernelContext* ctx) override {
    const TensorKey& key = ctx->input(1);
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result));
    result->scalar<bool>()() = map->tensors().find(key) != map->tensors().end();
  }
};

class TensorMapStackKeys : public OpKernel {
 public:
  explicit TensorMapStackKeys(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_dtype", &key_dtype_));
  }
  ~TensorMapStackKeys() override {}

  void Compute(OpKernelContext* ctx) override {
    const TensorMap* map = nullptr;
    OP_REQUIRES_OK(ctx, GetInputMap(ctx, 0, &map));

    OP_REQUIRES(ctx, map->size() != 0,
                errors::InvalidArgument(
                    "TensorMapStackKeys cannot be called on empty map."));

    auto it = map->tensors().begin();
    TensorShape output_shape = it->first.shape();
    output_shape.InsertDim(0, map->tensors().size());
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &result));

    int i = 0;
    size_t sz = map->tensors().size();
    TensorShape key_shape = it->first.shape();
    while (it != map->tensors().end() && i < sz) {
      OP_REQUIRES(
          ctx, it->first.dtype() == key_dtype_,
          errors::InvalidArgument("Key does not match requested dtype."));
      OP_REQUIRES(
          ctx, it->first.shape() == key_shape,
          errors::InvalidArgument("Keys must all have the same shape."));
      OP_REQUIRES_OK(ctx, batch_util::CopyElementToSlice(it->first, result, i));
      i++;
      it++;
    }
  }

 private:
  DataType key_dtype_;
};

template <typename Device>
Status TensorMapBinaryAdd(OpKernelContext* ctx, const TensorMap& a,
                          const TensorMap& b, TensorMap* out) {
  // Binary add returns a map containing the union of keys.
  // Values with keys in the intersection are added.
  out->tensors() = a.tensors();
  for (const std::pair<TensorKey, Tensor>& p : b.tensors()) {
    absl::flat_hash_map<TensorKey, Tensor>::iterator it =
        out->tensors().find(p.first);
    if (it != out->tensors().end()) {
      Tensor out_tensor;
      TF_RETURN_IF_ERROR(
          BinaryAddTensors<Device>(ctx, p.second, it->second, &out_tensor));
      it->second = out_tensor;
    } else {
      out->tensors().emplace(p.first, p.second);
    }
  }
  return OkStatus();
}

template <typename Device>
Status TensorMapZerosLike(OpKernelContext* ctx, const TensorMap& x,
                          TensorMap* y) {
  // Zeros like returns an empty map.
  return OkStatus();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAP_KERNELS_H_
