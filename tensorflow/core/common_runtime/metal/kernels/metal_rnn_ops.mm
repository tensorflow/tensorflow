/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// The single-step recurrent cells: LSTMBlockCell and GRUBlockCell, with their
// gradients.
//
// A cell is a matrix multiply followed by a handful of elementwise gates, so
// each one is a single MPSGraph, and the graph cache keys on the shapes. What
// makes these ops worth having on the device is not the arithmetic but the
// round trips they save: a cell that runs on the host forces the whole
// sequence's activations across the boundary on every step.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// One column range of a matrix, which is how every gate is addressed here.
MPSGraphTensor* Columns(MPSGraph* g, MPSGraphTensor* t, int64_t start,
                        int64_t length) {
  return [g sliceTensor:t
              dimension:1
                  start:static_cast<NSInteger>(start)
                 length:static_cast<NSInteger>(length)
                   name:nil];
}

MPSGraphTensor* Broadcast(MPSGraph* g, MPSGraphTensor* v, int64_t width) {
  return [g reshapeTensor:v
                withShape:@[ @1, @(static_cast<NSInteger>(width)) ]
                     name:nil];
}

MPSGraphTensor* Scalar(MPSGraph* g, double value) {
  return [g constantWithScalar:value dataType:MPSDataTypeFloat32];
}

MPSGraphTensor* OneMinus(MPSGraph* g, MPSGraphTensor* t) {
  return [g subtractionWithPrimaryTensor:Scalar(g, 1.0)
                         secondaryTensor:t
                                    name:nil];
}

MPSGraphTensor* Mul(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
  return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
}

MPSGraphTensor* Add(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
  return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
}

MPSGraphTensor* Sub(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
  return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
}

MPSGraphTensor* MatMul(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
  return [g matrixMultiplicationWithPrimaryTensor:a
                                  secondaryTensor:b
                                             name:nil];
}

MPSGraphTensor* Transpose(MPSGraph* g, MPSGraphTensor* t) {
  return [g transposeTensor:t dimension:0 withDimension:1 name:nil];
}

// Sum over the batch, leaving one value per unit.
MPSGraphTensor* SumOverBatch(MPSGraph* g, MPSGraphTensor* t, int64_t width) {
  return [g reshapeTensor:[g reductionSumWithTensor:t axes:@[ @0 ] name:nil]
                withShape:@[ @(static_cast<NSInteger>(width)) ]
                     name:nil];
}

struct RnnOp {
  float forget_bias = 1.0f;
  float cell_clip = 0.0f;
  bool use_peephole = false;
};

void* RnnOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new RnnOp();
  float value = 0.0f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "forget_bias", &value, status);
  if (TF_GetCode(status) == TF_OK) op->forget_bias = value;
  TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrFloat(ctx, "cell_clip", &value, status);
  if (TF_GetCode(status) == TF_OK) op->cell_clip = value;
  TF_SetStatus(status, TF_OK, "");
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "use_peephole", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->use_peephole = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void RnnOp_Delete(void* kernel) { delete static_cast<RnnOp*>(kernel); }

/*** LSTM BLOCK CELL ***/

// The gate layout differs between the original op and its V2: both pack four
// gates into one matrix, but V2 swaps the cell and forget columns.
struct GateLayout {
  bool ifco = false;
  int64_t i_offset(int64_t cell) const { return 0; }
  int64_t c_offset(int64_t cell) const { return ifco ? 2 * cell : cell; }
  int64_t f_offset(int64_t cell) const { return ifco ? cell : 2 * cell; }
  int64_t o_offset(int64_t cell) const { return 3 * cell; }
};

struct CellShapes {
  int64_t batch = 0;
  int64_t input_size = 0;
  int64_t cell = 0;

  std::vector<int64_t> x() const { return {batch, input_size}; }
  std::vector<int64_t> state() const { return {batch, cell}; }
  std::vector<int64_t> w() const { return {input_size + cell, 4 * cell}; }
  std::vector<int64_t> b() const { return {4 * cell}; }
  std::vector<int64_t> peep() const { return {cell}; }
  std::vector<int64_t> gates() const { return {batch, 4 * cell}; }
};

// One step of the cell. Inputs in order: x, cs_prev, h_prev, w, wci, wcf, wco,
// b. Outputs in order: i, cs, f, o, ci, co, h, which is the order the ops
// declare them in.
const CachedGraph* ForwardCellGraph(const RnnOp& op, GateLayout layout,
                                    const CellShapes& shapes,
                                    TF_Status* status) {
  const int64_t cell = shapes.cell;
  const std::vector<int64_t> x_shape = shapes.x();
  const std::vector<int64_t> state_shape = shapes.state();
  const std::vector<int64_t> w_shape = shapes.w();
  const std::vector<int64_t> b_shape = shapes.b();
  const std::vector<int64_t> p_shape = shapes.peep();

  std::string key = "LSTMCellForward";
  AppendShapeToKey(x_shape, &key);
  AppendShapeToKey(state_shape, &key);
  key.append("/fb").append(std::to_string(op.forget_bias));
  key.append("/cc").append(std::to_string(op.cell_clip));
  key.append(op.use_peephole ? "/peep" : "/plain");
  key.append(layout.ifco ? "/ifco" : "/icfo");

  const RnnOp captured = op;
  return LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(x_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* cs_prev = [g placeholderWithShape:MPSShape(state_shape)
                                                 dataType:MPSDataTypeFloat32
                                                     name:nil];
        MPSGraphTensor* h_prev = [g placeholderWithShape:MPSShape(state_shape)
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
        MPSGraphTensor* w = [g placeholderWithShape:MPSShape(w_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* wci = [g placeholderWithShape:MPSShape(p_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];
        MPSGraphTensor* wcf = [g placeholderWithShape:MPSShape(p_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];
        MPSGraphTensor* wco = [g placeholderWithShape:MPSShape(p_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];
        MPSGraphTensor* b = [g placeholderWithShape:MPSShape(b_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];

        // The concatenation is the whole reason w has input_size + cell rows.
        MPSGraphTensor* xh = [g concatTensor:x
                                  withTensor:h_prev
                                   dimension:1
                                        name:nil];
        MPSGraphTensor* gates =
            Add(g, MatMul(g, xh, w), Broadcast(g, b, 4 * cell));

        MPSGraphTensor* i_gate =
            Columns(g, gates, layout.i_offset(cell), cell);
        MPSGraphTensor* c_gate =
            Columns(g, gates, layout.c_offset(cell), cell);
        MPSGraphTensor* f_gate =
            Columns(g, gates, layout.f_offset(cell), cell);
        MPSGraphTensor* o_gate =
            Columns(g, gates, layout.o_offset(cell), cell);

        if (captured.use_peephole) {
          i_gate = Add(g, i_gate, Mul(g, cs_prev, Broadcast(g, wci, cell)));
        }
        MPSGraphTensor* i = [g sigmoidWithTensor:i_gate name:nil];
        MPSGraphTensor* ci = [g tanhWithTensor:c_gate name:nil];

        MPSGraphTensor* f_sum =
            Add(g, f_gate, Scalar(g, captured.forget_bias));
        if (captured.use_peephole) {
          f_sum = Add(g, f_sum, Mul(g, cs_prev, Broadcast(g, wcf, cell)));
        }
        MPSGraphTensor* f = [g sigmoidWithTensor:f_sum name:nil];

        MPSGraphTensor* cs = Add(g, Mul(g, i, ci), Mul(g, f, cs_prev));
        if (captured.cell_clip > 0.0f) {
          cs = [g clampWithTensor:cs
                   minValueTensor:Scalar(g, -captured.cell_clip)
                   maxValueTensor:Scalar(g, captured.cell_clip)
                             name:nil];
        }
        MPSGraphTensor* co = [g tanhWithTensor:cs name:nil];

        // The output gate's peephole reads the new state, not the old one.
        if (captured.use_peephole) {
          o_gate = Add(g, o_gate, Mul(g, cs, Broadcast(g, wco, cell)));
        }
        MPSGraphTensor* o = [g sigmoidWithTensor:o_gate name:nil];

        [out->inputs addObject:x];
        [out->inputs addObject:cs_prev];
        [out->inputs addObject:h_prev];
        [out->inputs addObject:w];
        [out->inputs addObject:wci];
        [out->inputs addObject:wcf];
        [out->inputs addObject:wco];
        [out->inputs addObject:b];
        [out->outputs addObject:i];
        [out->outputs addObject:cs];
        [out->outputs addObject:f];
        [out->outputs addObject:o];
        [out->outputs addObject:ci];
        [out->outputs addObject:co];
        [out->outputs addObject:Mul(g, o, co)];
      },
      status);
}

void LSTMCell_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  ScopedTensor in[8];
  for (int i = 0; i < 8; ++i) {
    TF_GetInput(ctx, i, in[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  const std::vector<int64_t> x_shape = ShapeOf(in[0].get());
  const std::vector<int64_t> cs_shape = ShapeOf(in[1].get());
  if (x_shape.size() != 2 || cs_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: LSTMBlockCell expects rank-2 inputs.");
    return;
  }
  CellShapes shapes;
  shapes.batch = x_shape[0];
  shapes.input_size = x_shape[1];
  shapes.cell = cs_shape[1];
  const std::vector<int64_t> state_shape = shapes.state();

  ScopedTensor outputs[7];
  for (int i = 0; i < 7; ++i) {
    outputs[i].reset(TF_AllocateOutput(
        ctx, i, TF_FLOAT, state_shape.data(), 2,
        static_cast<size_t>(shapes.batch * shapes.cell) * sizeof(float),
        status));
    if (TF_GetCode(status) != TF_OK) return;
  }
  if (shapes.batch * shapes.cell == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  const CachedGraph* cached =
      ForwardCellGraph(*op, GateLayout(), shapes, status);
  if (cached == nullptr) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  for (int i = 0; i < 8; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(in[i].get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [feeds addObject:data];
  }
  NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
  for (int i = 0; i < 7; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(outputs[i].get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [results addObject:data];
  }
  RunGraph(stream, *cached, feeds, results, status);
}

/*** LSTM BLOCK CELL GRADIENT ***/

void LSTMCellGrad_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx,
                              TF_Status* status) {
  ScopedTensor in[16];
  for (int i = 0; i < 16; ++i) {
    TF_GetInput(ctx, i, in[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  const std::vector<int64_t> x_shape = ShapeOf(in[0].get());
  const std::vector<int64_t> cell_shape = ShapeOf(in[1].get());
  if (x_shape.size() != 2 || cell_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: LSTMBlockCellGrad expects rank-2 inputs.");
    return;
  }
  const int64_t batch = cell_shape[0];
  const int64_t cell = cell_shape[1];
  const std::vector<int64_t> gates_shape = {batch, 4 * cell};
  const std::vector<int64_t> p_shape = {cell};

  ScopedTensor cs_prev_grad, dicfo, wci_grad, wcf_grad, wco_grad;
  cs_prev_grad.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, cell_shape.data(), 2,
      static_cast<size_t>(batch * cell) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  dicfo.reset(TF_AllocateOutput(
      ctx, 1, TF_FLOAT, gates_shape.data(), 2,
      static_cast<size_t>(batch * 4 * cell) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  ScopedTensor* peep[3] = {&wci_grad, &wcf_grad, &wco_grad};
  for (int i = 0; i < 3; ++i) {
    peep[i]->reset(TF_AllocateOutput(ctx, i + 2, TF_FLOAT, p_shape.data(), 1,
                                     static_cast<size_t>(cell) * sizeof(float),
                                     status));
    if (TF_GetCode(status) != TF_OK) return;
  }
  if (batch * cell == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "LSTMBlockCellGrad";
  AppendShapeToKey(cell_shape, &key);
  key.append(op->use_peephole ? "/peep" : "/plain");

  const bool use_peephole = op->use_peephole;
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        // Only the tensors the gradient actually reads are placeholders; x, w
        // and b are inputs of the op but do not appear in its formulas.
        MPSGraphTensor* cs_prev = [g placeholderWithShape:MPSShape(cell_shape)
                                                 dataType:MPSDataTypeFloat32
                                                     name:nil];
        MPSGraphTensor* wci = [g placeholderWithShape:MPSShape(p_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];
        MPSGraphTensor* wcf = [g placeholderWithShape:MPSShape(p_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];
        MPSGraphTensor* wco = [g placeholderWithShape:MPSShape(p_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];
        MPSGraphTensor* i = [g placeholderWithShape:MPSShape(cell_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* cs = [g placeholderWithShape:MPSShape(cell_shape)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
        MPSGraphTensor* f = [g placeholderWithShape:MPSShape(cell_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* o = [g placeholderWithShape:MPSShape(cell_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* ci = [g placeholderWithShape:MPSShape(cell_shape)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
        MPSGraphTensor* co = [g placeholderWithShape:MPSShape(cell_shape)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
        MPSGraphTensor* cs_grad = [g placeholderWithShape:MPSShape(cell_shape)
                                                 dataType:MPSDataTypeFloat32
                                                     name:nil];
        MPSGraphTensor* h_grad = [g placeholderWithShape:MPSShape(cell_shape)
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];

        MPSGraphTensor* do_ =
            Mul(g, Mul(g, Mul(g, o, OneMinus(g, o)), h_grad), co);
        MPSGraphTensor* dcs =
            Add(g, Mul(g, Mul(g, OneMinus(g, Mul(g, co, co)), h_grad), o),
                cs_grad);
        if (use_peephole) {
          dcs = Add(g, dcs, Mul(g, do_, Broadcast(g, wco, cell)));
        }
        MPSGraphTensor* dci =
            Mul(g, Mul(g, OneMinus(g, Mul(g, ci, ci)), dcs), i);
        MPSGraphTensor* df =
            Mul(g, Mul(g, Mul(g, f, OneMinus(g, f)), dcs), cs_prev);
        MPSGraphTensor* di =
            Mul(g, Mul(g, Mul(g, i, OneMinus(g, i)), dcs), ci);

        MPSGraphTensor* dgates = [g concatTensors:@[ di, dci, df, do_ ]
                                        dimension:1
                                             name:nil];
        MPSGraphTensor* cs_prev_g = Mul(g, dcs, f);
        MPSGraphTensor* wci_g;
        MPSGraphTensor* wcf_g;
        MPSGraphTensor* wco_g;
        if (use_peephole) {
          cs_prev_g =
              Add(g, cs_prev_g,
                  Add(g, Mul(g, di, Broadcast(g, wci, cell)),
                      Mul(g, df, Broadcast(g, wcf, cell))));
          wci_g = SumOverBatch(g, Mul(g, di, cs_prev), cell);
          wcf_g = SumOverBatch(g, Mul(g, df, cs_prev), cell);
          wco_g = SumOverBatch(g, Mul(g, do_, cs), cell);
        } else {
          // Without peepholes those weights do not exist, and TensorFlow
          // still emits their gradients, as zeros.
          MPSGraphTensor* zero = [g constantWithScalar:0.0
                                                 shape:MPSShape(p_shape)
                                              dataType:MPSDataTypeFloat32];
          wci_g = zero;
          wcf_g = zero;
          wco_g = zero;
        }

        [out->inputs addObject:cs_prev];
        [out->inputs addObject:wci];
        [out->inputs addObject:wcf];
        [out->inputs addObject:wco];
        [out->inputs addObject:i];
        [out->inputs addObject:cs];
        [out->inputs addObject:f];
        [out->inputs addObject:o];
        [out->inputs addObject:ci];
        [out->inputs addObject:co];
        [out->inputs addObject:cs_grad];
        [out->inputs addObject:h_grad];
        [out->outputs addObject:cs_prev_g];
        [out->outputs addObject:dgates];
        [out->outputs addObject:wci_g];
        [out->outputs addObject:wcf_g];
        [out->outputs addObject:wco_g];
      },
      status);
  if (cached == nullptr) return;

  // The graph's inputs in the order the builder added them.
  static constexpr int kFeedIndex[] = {1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14,
                                       15};
  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  for (int index : kFeedIndex) {
    MPSGraphTensorData* data =
        TensorDataForTensor(in[index].get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [feeds addObject:data];
  }
  ScopedTensor* outs[5] = {&cs_prev_grad, &dicfo, &wci_grad, &wcf_grad,
                           &wco_grad};
  NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
  for (int i = 0; i < 5; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(outs[i]->get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [results addObject:data];
  }
  RunGraph(stream, *cached, feeds, results, status);
}

/*** GRU BLOCK CELL ***/

void GRUCell_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor in[6];
  for (int i = 0; i < 6; ++i) {
    TF_GetInput(ctx, i, in[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  const std::vector<int64_t> x_shape = ShapeOf(in[0].get());
  const std::vector<int64_t> h_shape = ShapeOf(in[1].get());
  if (x_shape.size() != 2 || h_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: GRUBlockCell expects rank-2 inputs.");
    return;
  }
  const int64_t batch = x_shape[0];
  const int64_t input_size = x_shape[1];
  const int64_t cell = h_shape[1];
  const std::vector<int64_t> w_ru_shape = {input_size + cell, 2 * cell};
  const std::vector<int64_t> w_c_shape = {input_size + cell, cell};
  const std::vector<int64_t> b_ru_shape = {2 * cell};
  const std::vector<int64_t> b_c_shape = {cell};

  ScopedTensor outputs[4];
  for (int i = 0; i < 4; ++i) {
    outputs[i].reset(TF_AllocateOutput(
        ctx, i, TF_FLOAT, h_shape.data(), 2,
        static_cast<size_t>(batch * cell) * sizeof(float), status));
    if (TF_GetCode(status) != TF_OK) return;
  }
  if (batch * cell == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "GRUBlockCell";
  AppendShapeToKey(x_shape, &key);
  AppendShapeToKey(h_shape, &key);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(x_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* h_prev = [g placeholderWithShape:MPSShape(h_shape)
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
        MPSGraphTensor* w_ru = [g placeholderWithShape:MPSShape(w_ru_shape)
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        MPSGraphTensor* w_c = [g placeholderWithShape:MPSShape(w_c_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];
        MPSGraphTensor* b_ru = [g placeholderWithShape:MPSShape(b_ru_shape)
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        MPSGraphTensor* b_c = [g placeholderWithShape:MPSShape(b_c_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];

        MPSGraphTensor* xh = [g concatTensor:x
                                  withTensor:h_prev
                                   dimension:1
                                        name:nil];
        MPSGraphTensor* ru =
            Add(g, MatMul(g, xh, w_ru), Broadcast(g, b_ru, 2 * cell));
        MPSGraphTensor* r =
            [g sigmoidWithTensor:Columns(g, ru, 0, cell) name:nil];
        MPSGraphTensor* u =
            [g sigmoidWithTensor:Columns(g, ru, cell, cell) name:nil];

        // The reset gate multiplies the previous state before the candidate's
        // matrix multiply, which is why there are two concatenations.
        MPSGraphTensor* xhr = [g concatTensor:x
                                   withTensor:Mul(g, h_prev, r)
                                    dimension:1
                                         name:nil];
        MPSGraphTensor* c = [g
            tanhWithTensor:Add(g, MatMul(g, xhr, w_c), Broadcast(g, b_c, cell))
                      name:nil];
        // h = u * h_prev + (1 - u) * c, written as TensorFlow writes it.
        MPSGraphTensor* h = Add(g, Mul(g, u, Sub(g, h_prev, c)), c);

        [out->inputs addObject:x];
        [out->inputs addObject:h_prev];
        [out->inputs addObject:w_ru];
        [out->inputs addObject:w_c];
        [out->inputs addObject:b_ru];
        [out->inputs addObject:b_c];
        [out->outputs addObject:r];
        [out->outputs addObject:u];
        [out->outputs addObject:c];
        [out->outputs addObject:h];
      },
      status);
  if (cached == nullptr) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  for (int i = 0; i < 6; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(in[i].get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [feeds addObject:data];
  }
  NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
  for (int i = 0; i < 4; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(outputs[i].get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [results addObject:data];
  }
  RunGraph(stream, *cached, feeds, results, status);
}

/*** GRU BLOCK CELL GRADIENT ***/

void GRUCellGrad_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor in[10];
  for (int i = 0; i < 10; ++i) {
    TF_GetInput(ctx, i, in[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  const std::vector<int64_t> x_shape = ShapeOf(in[0].get());
  const std::vector<int64_t> h_shape = ShapeOf(in[1].get());
  if (x_shape.size() != 2 || h_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: GRUBlockCellGrad expects rank-2 inputs.");
    return;
  }
  const int64_t batch = x_shape[0];
  const int64_t input_size = x_shape[1];
  const int64_t cell = h_shape[1];
  const std::vector<int64_t> w_ru_shape = {input_size + cell, 2 * cell};
  const std::vector<int64_t> w_c_shape = {input_size + cell, cell};
  const std::vector<int64_t> ru_shape = {batch, 2 * cell};

  ScopedTensor d_x, d_h_prev, d_c_bar_out, d_ru_out;
  d_x.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, x_shape.data(), 2,
      static_cast<size_t>(batch * input_size) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  d_h_prev.reset(TF_AllocateOutput(
      ctx, 1, TF_FLOAT, h_shape.data(), 2,
      static_cast<size_t>(batch * cell) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  d_c_bar_out.reset(TF_AllocateOutput(
      ctx, 2, TF_FLOAT, h_shape.data(), 2,
      static_cast<size_t>(batch * cell) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  d_ru_out.reset(TF_AllocateOutput(
      ctx, 3, TF_FLOAT, ru_shape.data(), 2,
      static_cast<size_t>(batch * 2 * cell) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (batch * cell == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "GRUBlockCellGrad";
  AppendShapeToKey(x_shape, &key);
  AppendShapeToKey(h_shape, &key);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* h_prev = [g placeholderWithShape:MPSShape(h_shape)
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
        MPSGraphTensor* w_ru = [g placeholderWithShape:MPSShape(w_ru_shape)
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        MPSGraphTensor* w_c = [g placeholderWithShape:MPSShape(w_c_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];
        MPSGraphTensor* r = [g placeholderWithShape:MPSShape(h_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* u = [g placeholderWithShape:MPSShape(h_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* c = [g placeholderWithShape:MPSShape(h_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* d_h = [g placeholderWithShape:MPSShape(h_shape)
                                             dataType:MPSDataTypeFloat32
                                                 name:nil];

        MPSGraphTensor* d_c_bar =
            Mul(g, Mul(g, d_h, OneMinus(g, u)), OneMinus(g, Mul(g, c, c)));
        MPSGraphTensor* d_u_bar =
            Mul(g, Mul(g, Mul(g, d_h, Sub(g, h_prev, c)), u), OneMinus(g, u));

        // The candidate's gradient flows back through w_c into both x and the
        // reset-scaled state.
        MPSGraphTensor* comp2 = MatMul(g, d_c_bar, Transpose(g, w_c));
        MPSGraphTensor* d_hr =
            [g sliceTensor:comp2
                 dimension:1
                     start:static_cast<NSInteger>(input_size)
                    length:static_cast<NSInteger>(cell)
                      name:nil];
        MPSGraphTensor* d_r_bar =
            Mul(g, Mul(g, Mul(g, d_hr, h_prev), r), OneMinus(g, r));
        MPSGraphTensor* d_ru = [g concatTensor:d_r_bar
                                    withTensor:d_u_bar
                                     dimension:1
                                          name:nil];
        MPSGraphTensor* comp1 = MatMul(g, d_ru, Transpose(g, w_ru));

        MPSGraphTensor* d_x_t = [g sliceTensor:Add(g, comp1, comp2)
                                     dimension:1
                                         start:0
                                        length:static_cast<NSInteger>(
                                                   input_size)
                                          name:nil];
        MPSGraphTensor* d_h_prev_t =
            Add(g,
                [g sliceTensor:comp1
                     dimension:1
                         start:static_cast<NSInteger>(input_size)
                        length:static_cast<NSInteger>(cell)
                          name:nil],
                Add(g, Mul(g, d_hr, r), Mul(g, d_h, u)));

        [out->inputs addObject:h_prev];
        [out->inputs addObject:w_ru];
        [out->inputs addObject:w_c];
        [out->inputs addObject:r];
        [out->inputs addObject:u];
        [out->inputs addObject:c];
        [out->inputs addObject:d_h];
        [out->outputs addObject:d_x_t];
        [out->outputs addObject:d_h_prev_t];
        [out->outputs addObject:d_c_bar];
        [out->outputs addObject:d_ru];
      },
      status);
  if (cached == nullptr) return;

  static constexpr int kFeedIndex[] = {1, 2, 3, 6, 7, 8, 9};
  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  for (int index : kFeedIndex) {
    MPSGraphTensorData* data =
        TensorDataForTensor(in[index].get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [feeds addObject:data];
  }
  ScopedTensor* outs[4] = {&d_x, &d_h_prev, &d_c_bar_out, &d_ru_out};
  NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
  for (int i = 0; i < 4; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(outs[i]->get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [results addObject:data];
  }
  RunGraph(stream, *cached, feeds, results, status);
}

#define METAL_RNN_COMPUTE(NAME, IMPL)                                       \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<RnnOp*>(kernel);                                 \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a recurrent kernel has no state.");              \
    } else {                                                                \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_RNN_COMPUTE(LSTMCell_Compute, LSTMCell_ComputeImpl)
METAL_RNN_COMPUTE(LSTMCellGrad_Compute, LSTMCellGrad_ComputeImpl)
METAL_RNN_COMPUTE(GRUCell_Compute, GRUCell_ComputeImpl)
METAL_RNN_COMPUTE(GRUCellGrad_Compute, GRUCellGrad_ComputeImpl)

#undef METAL_RNN_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &RnnOp_Create, compute, &RnnOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalRnnKernels() {
  Register("LSTMBlockCell", &LSTMCell_Compute, "MetalLSTMBlockCell");
  Register("LSTMBlockCellGrad", &LSTMCellGrad_Compute,
           "MetalLSTMBlockCellGrad");
  Register("GRUBlockCell", &GRUCell_Compute, "MetalGRUBlockCell");
  Register("GRUBlockCellGrad", &GRUCellGrad_Compute, "MetalGRUBlockCellGrad");
}

}  // namespace metal
}  // namespace tensorflow
