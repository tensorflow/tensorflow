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
#include "tensorflow/core/framework/variant_encode_decode.h"

#include <iostream>
using namespace std;

namespace tensorflow {

Status GetInputMap(OpKernelContext* c, int index, const TensorMap** map) {
  if (!TensorShapeUtils::IsScalar(c->input(index).shape())) {
    return errors::InvalidArgument("Input list must be a scalar saw: ",
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
  // the `input_list` to it.
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
  explicit EmptyTensorMap(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    std::cout << "hello EmptyTensorMap map_kernels.h" << std::endl;
    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &result, attr));
    TensorMap empty;
    result->scalar<Variant>()() = std::move(empty);
  }

 private:
  DataType element_dtype_;
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

class TensorMapZeros : public OpKernel {
 public:
  explicit TensorMapZeros(OpKernelConstruction* c) : OpKernel(c) {
    //OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }
  ~TensorMapZeros() override {}

  void Compute(OpKernelContext* c) override {
    std::cout << "hello TensorMapInsert kernel" << std::endl;
    const Tensor& temp_key = c->input(1);
    const TensorKey key = TensorKey(temp_key);
    std::cout << "got key" << std::endl;
    const Tensor& value = c->input(2);
    std::cout << "got value" << std::endl;

    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));
    std::cout << "got map" << std::endl;
    //TensorMap output_map;
    //OP_REQUIRES_OK(c, ForwardInputOrCreateNewMap(c, 0, 0, *m, &output_map));
    //std::cout << "create output" << std::endl;
    //output_map = m->Zeros();
    //c->set_output(0, &&output_map);
    //std::cout << "inserted" << std::endl;

    Tensor* result;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape{}, &result, attr));
    TensorMap output_map = m->Zeros();
    result->scalar<Variant>()() = std::move(output_map);
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorMapZeros").Device(DEVICE_CPU),
                        TensorMapZeros);

class TensorMapInsert : public OpKernel {
 public:
  explicit TensorMapInsert(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }
  ~TensorMapInsert() override {}

  void Compute(OpKernelContext* c) override {
    std::cout << "hello TensorMapInsert kernel" << std::endl;
    const Tensor& temp_key = c->input(1);
    const TensorKey key = TensorKey(temp_key);
    std::cout << "got key" << std::endl;
    const Tensor& value = c->input(2);
    std::cout << "got value" << std::endl;
    /*OP_REQUIRES(c, element_dtype_ == value.dtype(),
                errors::InvalidArgument("Invalid data types; list elements ",
                                        DataTypeString(element_dtype_),
                                        " but tried to append ",
                                        DataTypeString(value.dtype())));*/

    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));
    std::cout << "got map" << std::endl;
    /*OP_REQUIRES(c, m->element_shape.IsCompatibleWith(input.shape()),
                errors::InvalidArgument(
                    "Tried to append a map with incompatible shape to a "
                    "list. Op element shape: ",
                    input.shape().DebugString(),
                    " list shape: ", m->element_shape.DebugString()));*/
    /*OP_REQUIRES(c, element_dtype_ == m->element_dtype,
                errors::InvalidArgument("Invalid data types; op elements ",
                                        DataTypeString(element_dtype_),
                                        " but list elements ",
                                        DataTypeString(l->element_dtype)));

    if (l->max_num_elements != -1) {
      OP_REQUIRES(
          c, l->tensors().size() < l->max_num_elements,
          errors::InvalidArgument("Tried to push item into a full list",
                                  " list size: ", l->tensors().size(),
                                  " max_num_elements: ", l->max_num_elements));
    }*/
    TensorMap* output_map = nullptr;
    OP_REQUIRES_OK(c, ForwardInputOrCreateNewMap(c, 0, 0, *m, &output_map));
    std::cout << "create output" << std::endl;
    output_map->insert(key, value);
    std::cout << "inserted" << std::endl;
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorMapInsert").Device(DEVICE_CPU),
                        TensorMapInsert);

class TensorMapLookup : public OpKernel {
 public:
  explicit TensorMapLookup(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }
  ~TensorMapLookup() override {}

  void Compute(OpKernelContext* c) override {
    std::cout << "hello TensorMapInsert kernel" << std::endl;
    const Tensor& temp_key = c->input(1);
    const TensorKey key = TensorKey(temp_key);
    std::cout << "got key" << std::endl;
    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));
    std::cout << "got map" << std::endl;
    c->set_output(0, m->tensors().find(key)->second);
    std::cout << "finished" << std::endl;
  }

 private:
  DataType element_dtype_;
};

class TensorMapErase : public OpKernel {
 public:
  explicit TensorMapErase(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }

  void Compute(OpKernelContext* c) override {
    std::cout << "hello TensorMapErase op" << std::endl;
    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));
    const Tensor& temp_key = c->input(1);
    const TensorKey key = TensorKey(temp_key);
    /*OP_REQUIRES(c, element_dtype_ == l->element_dtype,
                errors::InvalidArgument("Invalid data types; op elements ",
                                        DataTypeString(element_dtype_),
                                        " but list elements ",
                                        DataTypeString(l->element_dtype)));*/

    OP_REQUIRES(c, !m->tensors().empty(),
                errors::InvalidArgument("Trying to erase from an empty map."));

    OP_REQUIRES(c, m->tensors().find(key) != m->tensors().end(),
                errors::InvalidArgument("Trying to erase non-existent item."));

    const Tensor& t = m->tensors().find(key)->second;
    c->set_output(1, t);
    /*if (t.dtype() != DT_INVALID) {
      c->set_output(1, t);
    } else {
      PartialTensorShape partial_element_shape;
      OP_REQUIRES_OK(
          c, GetElementShapeFromInput(c, *l, 1, &partial_element_shape));
      TensorShape element_shape;
      OP_REQUIRES(
          c, partial_element_shape.AsTensorShape(&element_shape),
          errors::InvalidArgument("Trying to read an uninitialized tensor but ",
                                  "element_shape is not fully defined.",
                                  partial_element_shape.DebugString()));
      Tensor* result;
      AllocatorAttributes attr;
      if (element_dtype_ == DT_VARIANT) {
        attr.set_on_host(true);
      }
      OP_REQUIRES_OK(c, c->allocate_output(1, element_shape, &result, attr));
      functor::SetZeroFunctor<Device, T>()(c->eigen_device<Device>(),
                                           result->flat<T>());
    }*/

    TensorMap* output_map = nullptr;
    OP_REQUIRES_OK(c, ForwardInputOrCreateNewMap(c, 0, 0, *m, &output_map));
    output_map->tensors().erase(key);
  }

 private:
  DataType element_dtype_;
};

class TensorMapReplace : public OpKernel {
 public:
  explicit TensorMapReplace(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("element_dtype", &element_dtype_));
  }
  ~TensorMapReplace() override {}

  void Compute(OpKernelContext* c) override {
    std::cout << "hello TensorMapReplace kernel" << std::endl;
    const Tensor& temp_key = c->input(1);
    const TensorKey key = TensorKey(temp_key);
    std::cout << "got key" << std::endl;
    const Tensor& value = c->input(2);
    std::cout << "got value" << std::endl;
    /*OP_REQUIRES(c, element_dtype_ == value.dtype(),
                errors::InvalidArgument("Invalid data types; list elements ",
                                        DataTypeString(element_dtype_),
                                        " but tried to append ",
                                        DataTypeString(value.dtype())));*/

    const TensorMap* m = nullptr;
    OP_REQUIRES_OK(c, GetInputMap(c, 0, &m));
    std::cout << "got map" << std::endl;
    /*OP_REQUIRES(c, m->element_shape.IsCompatibleWith(input.shape()),
                errors::InvalidArgument(
                    "Tried to append a map with incompatible shape to a "
                    "list. Op element shape: ",
                    input.shape().DebugString(),
                    " list shape: ", m->element_shape.DebugString()));*/
    /*OP_REQUIRES(c, element_dtype_ == m->element_dtype,
                errors::InvalidArgument("Invalid data types; op elements ",
                                        DataTypeString(element_dtype_),
                                        " but list elements ",
                                        DataTypeString(l->element_dtype)));

    if (l->max_num_elements != -1) {
      OP_REQUIRES(
          c, l->tensors().size() < l->max_num_elements,
          errors::InvalidArgument("Tried to push item into a full list",
                                  " list size: ", l->tensors().size(),
                                  " max_num_elements: ", l->max_num_elements));
    }*/
    TensorMap* output_map = nullptr;
    OP_REQUIRES_OK(c, ForwardInputOrCreateNewMap(c, 0, 0, *m, &output_map));
    std::cout << "create output" << std::endl;
    output_map->replace(key,value);
    std::cout << "inserted" << std::endl;
  }

 private:
  DataType element_dtype_;
};

REGISTER_KERNEL_BUILDER(Name("TensorMapReplace").Device(DEVICE_CPU),
                        TensorMapReplace);

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    cout << "Hello World - Op" << endl;
    // Grab the input tensor
    const Tensor& input_tensor = c->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(c, c->allocate_output(0, input_tensor.shape(), 
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // Set all but the first element of the output tensor to 0
    const int N = input.size();
    for (int i=1; i<N; i++) {
      output_flat(i) = 0;
    }
    
    // Preserve the first input value if possible
    if (N > 0) output_flat(0) = input(0);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAP_KERNELS_H_
