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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

class TpuConversions {
 public:
  static stream_executor::DeviceMemoryBase
  SE_DeviceMemoryBaseToDeviceMemoryBase(SE_DeviceMemoryBase se_base) {
    stream_executor::DeviceMemoryBase base(se_base.opaque, se_base.size);
    base.SetPayload(se_base.payload);
    return base;
  }

  static SE_DeviceMemoryBase DeviceMemoryBaseToSE_DeviceMemoryBase(
      const stream_executor::DeviceMemoryBase& base) {
    SE_DeviceMemoryBase se_base;
    se_base.opaque = const_cast<void*>(base.opaque());
    se_base.payload = base.payload();
    se_base.size = base.size();
    return se_base;
  }

  static xla::Shape CShapeToXlaShape(XLA_Shape* shape) {
    xla::ShapeProto p;
    p.ParseFromArray(shape->bytes, shape->size);
    return xla::Shape(p);
  }

  static void XlaShapeToCShape(const xla::Shape& xla_shape,
                               XLA_Shape* c_shape) {
    xla::ShapeProto p = xla_shape.ToProto();
    std::string p_str = p.SerializeAsString();
    c_shape->bytes = new char[p_str.size()];
    c_shape->size = p_str.size();
    memcpy(c_shape->bytes, p_str.data(), p_str.size());
  }

  static void XLAShapedBufferToCShapedBuffer(
      const xla::ShapedBuffer& buffer, XLA_ShapedBuffer* c_device_buffer) {
    XlaShapeToCShape(buffer.on_host_shape(), &c_device_buffer->on_host_shape);
    XlaShapeToCShape(buffer.on_device_shape(),
                     &c_device_buffer->on_device_shape);
    c_device_buffer->device_ordinal = buffer.device_ordinal();
    absl::InlinedVector<SE_DeviceMemoryBase, 2> bases;
    for (auto& pair : buffer.buffers()) {
      bases.push_back(DeviceMemoryBaseToSE_DeviceMemoryBase(pair.second));
    }
    c_device_buffer->count = bases.size();
    c_device_buffer->bases = new SE_DeviceMemoryBase[bases.size()];
    for (int i = 0; i < bases.size(); ++i) {
      c_device_buffer->bases[i] = bases[i];
    }
  }

  static void XLALiteralToCLiteral(const xla::LiteralSlice& literal,
                                   XLA_Literal* c_literal) {
    XlaShapeToCShape(literal.shape(), &c_literal->shape);
    auto shapes = xla::ShapeUtil::GetLeafShapes(literal.shape());
    c_literal->buffers = new char*[shapes.size()];
    c_literal->sizes = new size_t[shapes.size()];
    c_literal->count = shapes.size();
    for (int i = 0; i < shapes.size(); ++i) {
      c_literal->buffers[i] = reinterpret_cast<char*>(
          const_cast<void*>(literal.untyped_data(shapes[i].index)));
      c_literal->sizes[i] = literal.size_bytes(shapes[i].index);
    }
  }

  static xla::MutableBorrowingLiteral CLiteralToXLALiteral(
      XLA_Literal* c_literal) {
    xla::Shape shape = CShapeToXlaShape(&c_literal->shape);
    LOG(INFO) << "Shape: " << shape.DebugString();
    return xla::MutableBorrowingLiteral(
        absl::MakeSpan(c_literal->buffers, c_literal->count), shape);
  }

  static void CShapeCleanup(XLA_Shape* c_shape) { delete[] c_shape->bytes; }

  static void CLiteralCleanup(XLA_Literal* c_literal) {
    delete[] c_literal->buffers;
    delete[] c_literal->sizes;
    CShapeCleanup(&c_literal->shape);
  }

  static void CShapedBufferCleanup(XLA_ShapedBuffer* c_buffer) {
    CShapeCleanup(&c_buffer->on_device_shape);
    CShapeCleanup(&c_buffer->on_host_shape);
    delete[] c_buffer->bases;
  }
};

#endif  // THIRD_PARTY_TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_
