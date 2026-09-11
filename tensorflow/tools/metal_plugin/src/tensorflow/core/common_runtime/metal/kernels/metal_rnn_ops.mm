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

#include <algorithm>
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

// The V2 ops have no forget_bias attribute: their forget bias is zero, and
// reading a missing attribute would silently leave the V1 default of one in
// place, which shifts every forget gate.
void* RnnOpV2_Create(TF_OpKernelConstruction* ctx) {
  void* kernel = RnnOp_Create(ctx);
  if (kernel != nullptr) static_cast<RnnOp*>(kernel)->forget_bias = 0.0f;
  return kernel;
}


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

/*** BLOCK LSTM: THE SEQUENCE FORMS ***/

// BlockLSTM runs the same cell over a whole sequence. The loop is on the host
// and each step is a graph run, which is what the shape of the problem allows:
// step t depends on step t-1, so there is nothing to parallelise across time,
// and the alternative would be a single graph with the length unrolled into
// it, recompiled for every sequence length a model sees.
//
// Nothing crosses to the host inside the loop. Each step reads its previous
// state straight out of the output tensors, at the byte offset of the previous
// step, and writes into the current one. That is the same aliasing trick the
// whole backend rests on: an MPSNDArray over an existing buffer at an offset.

// One time step's slice of a [T, ...] tensor.
BufferSlice StepSlice(const BufferSlice& base, int64_t step,
                      int64_t elements_per_step) {
  BufferSlice slice = base;
  slice.offset += static_cast<size_t>(step) * elements_per_step * sizeof(float);
  slice.length = static_cast<size_t>(elements_per_step) * sizeof(float);
  return slice;
}

// Reads the scalar that bounds the loop.
bool ReadSeqLenMax(TF_Tensor* t, int64_t* out, TF_Status* status) {
  const void* data = TF_TensorData(t);
  if (data == nullptr || TF_TensorElementCount(t) < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: seq_len_max must be a scalar in host memory.");
    return false;
  }
  *out = TF_TensorType(t) == TF_INT32
             ? static_cast<const int32_t*>(data)[0]
             : static_cast<const int64_t*>(data)[0];
  return true;
}

// Fills a whole tensor with zeros on the stream.
bool ZeroTensor(SP_Stream stream, TF_Tensor* tensor, TF_Status* status) {
  BufferSlice slice;
  if (!SliceForTensor(tensor, &slice, status)) return false;
  const size_t bytes = TF_TensorByteSize(tensor);
  if (bytes == 0) return true;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer to zero a tensor.");
    return false;
  }
  id<MTLBlitCommandEncoder> encoder =
      [command_buffer.get() blitCommandEncoder];
  [encoder fillBuffer:slice.buffer
                range:NSMakeRange(slice.offset, bytes)
                value:0];
  [encoder endEncoding];
  command_buffer.Commit();
  return true;
}

void BlockLSTM_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx,
                           GateLayout layout, TF_Status* status) {
  ScopedTensor in[9];
  for (int i = 0; i < 9; ++i) {
    TF_GetInput(ctx, i, in[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  int64_t seq_len_max = 0;
  if (!ReadSeqLenMax(in[0].get(), &seq_len_max, status)) return;

  const std::vector<int64_t> x_shape = ShapeOf(in[1].get());
  const std::vector<int64_t> state_shape = ShapeOf(in[2].get());
  if (x_shape.size() != 3 || state_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: BlockLSTM expects a rank-3 input and rank-2 state.");
    return;
  }
  const int64_t time_steps = x_shape[0];
  CellShapes shapes;
  shapes.batch = x_shape[1];
  shapes.input_size = x_shape[2];
  shapes.cell = state_shape[1];
  const int64_t per_step = shapes.batch * shapes.cell;
  const int64_t x_per_step = shapes.batch * shapes.input_size;
  const std::vector<int64_t> out_shape = {time_steps, shapes.batch,
                                          shapes.cell};

  ScopedTensor outputs[7];
  for (int i = 0; i < 7; ++i) {
    outputs[i].reset(TF_AllocateOutput(
        ctx, i, TF_FLOAT, out_shape.data(), 3,
        static_cast<size_t>(time_steps * per_step) * sizeof(float), status));
    if (TF_GetCode(status) != TF_OK) return;
  }
  if (time_steps * per_step == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  // Steps past seq_len_max are defined to be zero, and zeroing up front also
  // means an early exit leaves no uninitialised memory behind.
  for (int i = 0; i < 7; ++i) {
    if (!ZeroTensor(stream, outputs[i].get(), status)) return;
  }
  const int64_t steps = std::min(time_steps, std::max<int64_t>(seq_len_max, 0));
  if (steps == 0) return;

  const CachedGraph* cached = ForwardCellGraph(*op, layout, shapes, status);
  if (cached == nullptr) return;

  BufferSlice x_slice, w_slice, wci_slice, wcf_slice, wco_slice, b_slice;
  BufferSlice cs_prev_slice, h_prev_slice;
  if (!SliceForTensor(in[1].get(), &x_slice, status)) return;
  if (!SliceForTensor(in[2].get(), &cs_prev_slice, status)) return;
  if (!SliceForTensor(in[3].get(), &h_prev_slice, status)) return;
  if (!SliceForTensor(in[4].get(), &w_slice, status)) return;
  if (!SliceForTensor(in[5].get(), &wci_slice, status)) return;
  if (!SliceForTensor(in[6].get(), &wcf_slice, status)) return;
  if (!SliceForTensor(in[7].get(), &wco_slice, status)) return;
  if (!SliceForTensor(in[8].get(), &b_slice, status)) return;
  BufferSlice out_slices[7];
  for (int i = 0; i < 7; ++i) {
    if (!SliceForTensor(outputs[i].get(), &out_slices[i], status)) return;
  }

  const std::vector<int64_t> step_x_shape = shapes.x();
  const std::vector<int64_t> step_state_shape = shapes.state();
  for (int64_t t = 0; t < steps; ++t) {
    // The previous state is the previous step's own output, read in place.
    const BufferSlice cs_prev =
        t == 0 ? cs_prev_slice : StepSlice(out_slices[1], t - 1, per_step);
    const BufferSlice h_prev =
        t == 0 ? h_prev_slice : StepSlice(out_slices[6], t - 1, per_step);

    NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
    const BufferSlice feed_slices[8] = {
        StepSlice(x_slice, t, x_per_step),
        cs_prev,
        h_prev,
        w_slice,
        wci_slice,
        wcf_slice,
        wco_slice,
        b_slice};
    const std::vector<int64_t> feed_shapes[8] = {
        step_x_shape, step_state_shape, step_state_shape, shapes.w(),
        shapes.peep(), shapes.peep(),   shapes.peep(),    shapes.b()};
    for (int i = 0; i < 8; ++i) {
      MPSGraphTensorData* data = TensorDataFor(feed_slices[i], feed_shapes[i],
                                               TF_FLOAT, device, status);
      if (data == nil) return;
      [feeds addObject:data];
    }
    NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
    for (int i = 0; i < 7; ++i) {
      MPSGraphTensorData* data =
          TensorDataFor(StepSlice(out_slices[i], t, per_step),
                        step_state_shape, TF_FLOAT, device, status);
      if (data == nil) return;
      [results addObject:data];
    }
    if (!RunGraph(stream, *cached, feeds, results, status)) return;
  }
}

// One step of the backward pass, including the projection of the gate
// gradients back through w. Inputs in order: w, wci, wcf, wco, cs_prev, i, cs,
// f, o, ci, co, cs_grad, h_grad, carry_cs, carry_h. Outputs in order: dgates,
// cs_prev_grad, h_prev_grad, x_grad.
const CachedGraph* BackwardStepGraph(const RnnOp& op, GateLayout layout,
                                     const CellShapes& shapes,
                                     TF_Status* status) {
  const int64_t cell = shapes.cell;
  const int64_t input_size = shapes.input_size;
  const std::vector<int64_t> state_shape = shapes.state();
  const std::vector<int64_t> w_shape = shapes.w();
  const std::vector<int64_t> p_shape = shapes.peep();

  std::string key = "LSTMSeqBackwardStep";
  AppendShapeToKey(shapes.x(), &key);
  AppendShapeToKey(state_shape, &key);
  key.append(op.use_peephole ? "/peep" : "/plain");
  key.append(layout.ifco ? "/ifco" : "/icfo");

  const bool use_peephole = op.use_peephole;
  return LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
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
        MPSGraphTensor* state[10];
        for (int i = 0; i < 10; ++i) {
          state[i] = [g placeholderWithShape:MPSShape(state_shape)
                                    dataType:MPSDataTypeFloat32
                                        name:nil];
        }
        MPSGraphTensor* cs_prev = state[0];
        MPSGraphTensor* i = state[1];
        MPSGraphTensor* cs = state[2];
        MPSGraphTensor* f = state[3];
        MPSGraphTensor* o = state[4];
        MPSGraphTensor* ci = state[5];
        MPSGraphTensor* co = state[6];
        MPSGraphTensor* cs_grad_in = state[7];
        MPSGraphTensor* h_grad_in = state[8];
        MPSGraphTensor* carry_cs = [g placeholderWithShape:MPSShape(state_shape)
                                                  dataType:MPSDataTypeFloat32
                                                      name:nil];
        MPSGraphTensor* carry_h = state[9];

        // The gradient arriving at this step is what the loss sends directly
        // plus what the next step sent back.
        MPSGraphTensor* h_grad = Add(g, h_grad_in, carry_h);
        MPSGraphTensor* cs_grad = Add(g, cs_grad_in, carry_cs);

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

        NSArray<MPSGraphTensor*>* ordered =
            layout.ifco ? @[ di, df, dci, do_ ] : @[ di, dci, df, do_ ];
        MPSGraphTensor* dgates = [g concatTensors:ordered
                                        dimension:1
                                             name:nil];
        MPSGraphTensor* cs_prev_grad = Mul(g, dcs, f);
        if (use_peephole) {
          cs_prev_grad =
              Add(g, cs_prev_grad,
                  Add(g, Mul(g, di, Broadcast(g, wci, cell)),
                      Mul(g, df, Broadcast(g, wcf, cell))));
        }
        // The gate gradients project back through w onto [x, h_prev].
        MPSGraphTensor* xh_grad = MatMul(g, dgates, Transpose(g, w));
        MPSGraphTensor* x_grad =
            [g sliceTensor:xh_grad
                 dimension:1
                     start:0
                    length:static_cast<NSInteger>(input_size)
                      name:nil];
        MPSGraphTensor* h_prev_grad =
            [g sliceTensor:xh_grad
                 dimension:1
                     start:static_cast<NSInteger>(input_size)
                    length:static_cast<NSInteger>(cell)
                      name:nil];

        [out->inputs addObject:w];
        [out->inputs addObject:wci];
        [out->inputs addObject:wcf];
        [out->inputs addObject:wco];
        for (int k = 0; k < 9; ++k) [out->inputs addObject:state[k]];
        [out->inputs addObject:carry_cs];
        [out->inputs addObject:carry_h];
        [out->outputs addObject:dgates];
        [out->outputs addObject:cs_prev_grad];
        [out->outputs addObject:h_prev_grad];
        [out->outputs addObject:x_grad];
      },
      status);
}

// Everything the loop does not need to do step by step: the weight, bias and
// peephole gradients are sums over the whole sequence, so they are one matrix
// multiply and three reductions after the loop rather than an accumulation
// inside it.
const CachedGraph* BackwardReduceGraph(const RnnOp& op, GateLayout layout,
                                       const CellShapes& shapes,
                                       int64_t time_steps, TF_Status* status) {
  const int64_t cell = shapes.cell;
  const int64_t batch = shapes.batch;
  const int64_t input_size = shapes.input_size;
  const int64_t rows = time_steps * batch;
  const std::vector<int64_t> x_all_shape = {time_steps, batch, input_size};
  const std::vector<int64_t> state_all_shape = {time_steps, batch, cell};
  const std::vector<int64_t> gates_all_shape = {time_steps, batch, 4 * cell};
  const std::vector<int64_t> state_shape = shapes.state();
  const std::vector<int64_t> p_shape = shapes.peep();

  std::string key = "LSTMSeqBackwardReduce";
  AppendShapeToKey(x_all_shape, &key);
  AppendShapeToKey(state_all_shape, &key);
  key.append(op.use_peephole ? "/peep" : "/plain");
  key.append(layout.ifco ? "/ifco" : "/icfo");

  const bool use_peephole = op.use_peephole;
  return LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x_all = [g placeholderWithShape:MPSShape(x_all_shape)
                                               dataType:MPSDataTypeFloat32
                                                   name:nil];
        MPSGraphTensor* h_all =
            [g placeholderWithShape:MPSShape(state_all_shape)
                           dataType:MPSDataTypeFloat32
                               name:nil];
        MPSGraphTensor* h_prev = [g placeholderWithShape:MPSShape(state_shape)
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
        MPSGraphTensor* cs_all =
            [g placeholderWithShape:MPSShape(state_all_shape)
                           dataType:MPSDataTypeFloat32
                               name:nil];
        MPSGraphTensor* cs_prev = [g placeholderWithShape:MPSShape(state_shape)
                                                 dataType:MPSDataTypeFloat32
                                                     name:nil];
        MPSGraphTensor* dgates =
            [g placeholderWithShape:MPSShape(gates_all_shape)
                           dataType:MPSDataTypeFloat32
                               name:nil];

        // The state a step consumed is the previous step's output, so both
        // state sequences are shifted by one and opened with the initial
        // state.
        NSArray<NSNumber*>* one_step = @[
          @1, @(static_cast<NSInteger>(batch)), @(static_cast<NSInteger>(cell))
        ];
        MPSGraphTensor* h_head = [g reshapeTensor:h_prev
                                        withShape:one_step
                                             name:nil];
        MPSGraphTensor* cs_head = [g reshapeTensor:cs_prev
                                         withShape:one_step
                                              name:nil];
        MPSGraphTensor* h_shift = h_head;
        MPSGraphTensor* cs_shift = cs_head;
        if (time_steps > 1) {
          h_shift = [g concatTensor:h_head
                         withTensor:[g sliceTensor:h_all
                                         dimension:0
                                             start:0
                                            length:static_cast<NSInteger>(
                                                       time_steps - 1)
                                              name:nil]
                          dimension:0
                               name:nil];
          cs_shift = [g concatTensor:cs_head
                          withTensor:[g sliceTensor:cs_all
                                          dimension:0
                                              start:0
                                             length:static_cast<NSInteger>(
                                                        time_steps - 1)
                                               name:nil]
                           dimension:0
                                name:nil];
        }

        NSArray<NSNumber*>* rows_by_input = @[
          @(static_cast<NSInteger>(rows)),
          @(static_cast<NSInteger>(input_size))
        ];
        NSArray<NSNumber*>* rows_by_cell = @[
          @(static_cast<NSInteger>(rows)), @(static_cast<NSInteger>(cell))
        ];
        NSArray<NSNumber*>* rows_by_gates = @[
          @(static_cast<NSInteger>(rows)), @(static_cast<NSInteger>(4 * cell))
        ];
        MPSGraphTensor* xh = [g
            concatTensor:[g reshapeTensor:x_all withShape:rows_by_input name:nil]
              withTensor:[g reshapeTensor:h_shift
                                withShape:rows_by_cell
                                     name:nil]
               dimension:1
                    name:nil];
        MPSGraphTensor* dg = [g reshapeTensor:dgates
                                    withShape:rows_by_gates
                                         name:nil];
        MPSGraphTensor* w_grad = MatMul(g, Transpose(g, xh), dg);
        MPSGraphTensor* b_grad = SumOverBatch(g, dg, 4 * cell);

        MPSGraphTensor* wci_grad;
        MPSGraphTensor* wcf_grad;
        MPSGraphTensor* wco_grad;
        if (use_peephole) {
          MPSGraphTensor* cs_shift_flat = [g reshapeTensor:cs_shift
                                                 withShape:rows_by_cell
                                                      name:nil];
          MPSGraphTensor* cs_flat = [g reshapeTensor:cs_all
                                           withShape:rows_by_cell
                                                name:nil];
          MPSGraphTensor* di = Columns(g, dg, layout.i_offset(cell), cell);
          MPSGraphTensor* df = Columns(g, dg, layout.f_offset(cell), cell);
          MPSGraphTensor* do_ = Columns(g, dg, layout.o_offset(cell), cell);
          wci_grad = SumOverBatch(g, Mul(g, di, cs_shift_flat), cell);
          wcf_grad = SumOverBatch(g, Mul(g, df, cs_shift_flat), cell);
          wco_grad = SumOverBatch(g, Mul(g, do_, cs_flat), cell);
        } else {
          MPSGraphTensor* zero = [g constantWithScalar:0.0
                                                 shape:MPSShape(p_shape)
                                              dataType:MPSDataTypeFloat32];
          wci_grad = zero;
          wcf_grad = zero;
          wco_grad = zero;
        }

        [out->inputs addObject:x_all];
        [out->inputs addObject:h_all];
        [out->inputs addObject:h_prev];
        [out->inputs addObject:cs_all];
        [out->inputs addObject:cs_prev];
        [out->inputs addObject:dgates];
        [out->outputs addObject:w_grad];
        [out->outputs addObject:b_grad];
        [out->outputs addObject:wci_grad];
        [out->outputs addObject:wcf_grad];
        [out->outputs addObject:wco_grad];
      },
      status);
}

void BlockLSTMGrad_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx,
                               GateLayout layout, TF_Status* status) {
  ScopedTensor in[18];
  for (int i = 0; i < 18; ++i) {
    TF_GetInput(ctx, i, in[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  int64_t seq_len_max = 0;
  if (!ReadSeqLenMax(in[0].get(), &seq_len_max, status)) return;

  const std::vector<int64_t> x_shape = ShapeOf(in[1].get());
  const std::vector<int64_t> state_shape_in = ShapeOf(in[2].get());
  if (x_shape.size() != 3 || state_shape_in.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: BlockLSTMGrad expects a rank-3 input and rank-2 "
                 "state.");
    return;
  }
  const int64_t time_steps = x_shape[0];
  CellShapes shapes;
  shapes.batch = x_shape[1];
  shapes.input_size = x_shape[2];
  shapes.cell = state_shape_in[1];
  const int64_t per_step = shapes.batch * shapes.cell;
  const int64_t x_per_step = shapes.batch * shapes.input_size;
  const int64_t gates_per_step = shapes.batch * 4 * shapes.cell;
  const std::vector<int64_t> state_shape = shapes.state();
  const std::vector<int64_t> gates_shape = shapes.gates();
  const std::vector<int64_t> w_shape = shapes.w();
  const std::vector<int64_t> b_shape = shapes.b();
  const std::vector<int64_t> p_shape = shapes.peep();
  const std::vector<int64_t> gates_all_shape = {time_steps, shapes.batch,
                                                4 * shapes.cell};

  ScopedTensor x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
      wco_grad, b_grad;
  x_grad.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, x_shape.data(), 3,
      static_cast<size_t>(time_steps * x_per_step) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  cs_prev_grad.reset(TF_AllocateOutput(
      ctx, 1, TF_FLOAT, state_shape.data(), 2,
      static_cast<size_t>(per_step) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  h_prev_grad.reset(TF_AllocateOutput(
      ctx, 2, TF_FLOAT, state_shape.data(), 2,
      static_cast<size_t>(per_step) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  w_grad.reset(TF_AllocateOutput(
      ctx, 3, TF_FLOAT, w_shape.data(), 2,
      static_cast<size_t>(w_shape[0] * w_shape[1]) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  ScopedTensor* peep[3] = {&wci_grad, &wcf_grad, &wco_grad};
  for (int i = 0; i < 3; ++i) {
    peep[i]->reset(TF_AllocateOutput(
        ctx, i + 4, TF_FLOAT, p_shape.data(), 1,
        static_cast<size_t>(shapes.cell) * sizeof(float), status));
    if (TF_GetCode(status) != TF_OK) return;
  }
  b_grad.reset(TF_AllocateOutput(
      ctx, 7, TF_FLOAT, b_shape.data(), 1,
      static_cast<size_t>(4 * shapes.cell) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (time_steps * per_step == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  // The gate gradients for the whole sequence, which the reduction consumes.
  // Steps past seq_len_max stay zero and so contribute nothing to the sums.
  ScopedTensor dgates_all;
  dgates_all.reset(TF_AllocateTemp(
      ctx, TF_FLOAT, gates_all_shape.data(), 3, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return;
  if (!ZeroTensor(stream, dgates_all.get(), status)) return;
  if (!ZeroTensor(stream, x_grad.get(), status)) return;

  // Two buffers for each carried gradient, alternated so a step never reads
  // and writes the same storage.
  ScopedTensor carry_cs[2], carry_h[2];
  for (int i = 0; i < 2; ++i) {
    carry_cs[i].reset(TF_AllocateTemp(ctx, TF_FLOAT, state_shape.data(), 2,
                                      nullptr, status));
    if (TF_GetCode(status) != TF_OK) return;
    carry_h[i].reset(TF_AllocateTemp(ctx, TF_FLOAT, state_shape.data(), 2,
                                     nullptr, status));
    if (TF_GetCode(status) != TF_OK) return;
    if (!ZeroTensor(stream, carry_cs[i].get(), status)) return;
    if (!ZeroTensor(stream, carry_h[i].get(), status)) return;
  }

  const int64_t steps = std::min(time_steps, std::max<int64_t>(seq_len_max, 0));
  if (steps > 0) {
    const CachedGraph* step_graph =
        BackwardStepGraph(*op, layout, shapes, status);
    if (step_graph == nullptr) return;

    BufferSlice w_s, wci_s, wcf_s, wco_s, cs_prev_s, x_grad_s, dgates_s;
    if (!SliceForTensor(in[4].get(), &w_s, status)) return;
    if (!SliceForTensor(in[5].get(), &wci_s, status)) return;
    if (!SliceForTensor(in[6].get(), &wcf_s, status)) return;
    if (!SliceForTensor(in[7].get(), &wco_s, status)) return;
    if (!SliceForTensor(in[2].get(), &cs_prev_s, status)) return;
    if (!SliceForTensor(x_grad.get(), &x_grad_s, status)) return;
    if (!SliceForTensor(dgates_all.get(), &dgates_s, status)) return;
    // i, cs, f, o, ci, co, cs_grad, h_grad, in the op's input order.
    static constexpr int kSeqInput[8] = {9, 10, 11, 12, 13, 14, 16, 17};
    BufferSlice seq_slices[8];
    for (int i = 0; i < 8; ++i) {
      if (!SliceForTensor(in[kSeqInput[i]].get(), &seq_slices[i], status)) {
        return;
      }
    }
    BufferSlice cs_all_s = seq_slices[1];
    BufferSlice carry_cs_s[2], carry_h_s[2];
    for (int i = 0; i < 2; ++i) {
      if (!SliceForTensor(carry_cs[i].get(), &carry_cs_s[i], status)) return;
      if (!SliceForTensor(carry_h[i].get(), &carry_h_s[i], status)) return;
    }
    BufferSlice cs_prev_grad_s, h_prev_grad_s;
    if (!SliceForTensor(cs_prev_grad.get(), &cs_prev_grad_s, status)) return;
    if (!SliceForTensor(h_prev_grad.get(), &h_prev_grad_s, status)) return;

    int carry = 0;
    for (int64_t t = steps - 1; t >= 0; --t) {
      NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
      const BufferSlice fixed[4] = {w_s, wci_s, wcf_s, wco_s};
      const std::vector<int64_t> fixed_shapes[4] = {w_shape, p_shape, p_shape,
                                                    p_shape};
      for (int i = 0; i < 4; ++i) {
        MPSGraphTensorData* data = TensorDataFor(fixed[i], fixed_shapes[i],
                                                 TF_FLOAT, device, status);
        if (data == nil) return;
        [feeds addObject:data];
      }
      // The state this step consumed is the previous step's output.
      const BufferSlice cs_prev_for_step =
          t == 0 ? cs_prev_s : StepSlice(cs_all_s, t - 1, per_step);
      MPSGraphTensorData* cs_prev_data = TensorDataFor(
          cs_prev_for_step, state_shape, TF_FLOAT, device, status);
      if (cs_prev_data == nil) return;
      [feeds addObject:cs_prev_data];
      for (int i = 0; i < 8; ++i) {
        MPSGraphTensorData* data =
            TensorDataFor(StepSlice(seq_slices[i], t, per_step), state_shape,
                          TF_FLOAT, device, status);
        if (data == nil) return;
        [feeds addObject:data];
      }
      MPSGraphTensorData* carry_cs_data = TensorDataFor(
          carry_cs_s[carry], state_shape, TF_FLOAT, device, status);
      if (carry_cs_data == nil) return;
      MPSGraphTensorData* carry_h_data = TensorDataFor(
          carry_h_s[carry], state_shape, TF_FLOAT, device, status);
      if (carry_h_data == nil) return;
      [feeds addObject:carry_cs_data];
      [feeds addObject:carry_h_data];

      // The last step of the loop is step zero, whose carried gradients are
      // exactly the op's cs_prev and h_prev gradients, so it writes them
      // straight into the outputs.
      const BufferSlice next_cs =
          t == 0 ? cs_prev_grad_s : carry_cs_s[1 - carry];
      const BufferSlice next_h = t == 0 ? h_prev_grad_s : carry_h_s[1 - carry];

      NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
      MPSGraphTensorData* dgates_data =
          TensorDataFor(StepSlice(dgates_s, t, gates_per_step), gates_shape,
                        TF_FLOAT, device, status);
      if (dgates_data == nil) return;
      MPSGraphTensorData* next_cs_data =
          TensorDataFor(next_cs, state_shape, TF_FLOAT, device, status);
      if (next_cs_data == nil) return;
      MPSGraphTensorData* next_h_data =
          TensorDataFor(next_h, state_shape, TF_FLOAT, device, status);
      if (next_h_data == nil) return;
      MPSGraphTensorData* x_grad_data =
          TensorDataFor(StepSlice(x_grad_s, t, x_per_step), shapes.x(),
                        TF_FLOAT, device, status);
      if (x_grad_data == nil) return;
      [results addObject:dgates_data];
      [results addObject:next_cs_data];
      [results addObject:next_h_data];
      [results addObject:x_grad_data];

      if (!RunGraph(stream, *step_graph, feeds, results, status)) return;
      carry = 1 - carry;
    }
  } else {
    // Nothing ran, so the carried gradients are the zeros they started as.
    if (!ZeroTensor(stream, cs_prev_grad.get(), status)) return;
    if (!ZeroTensor(stream, h_prev_grad.get(), status)) return;
  }

  const CachedGraph* reduce =
      BackwardReduceGraph(*op, layout, shapes, time_steps, status);
  if (reduce == nullptr) return;

  // x, h, h_prev, cs, cs_prev, dgates.
  static constexpr int kReduceInput[5] = {1, 15, 3, 10, 2};
  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  for (int i = 0; i < 5; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(in[kReduceInput[i]].get(), TF_FLOAT, device,
                            status);
    if (data == nil) return;
    [feeds addObject:data];
  }
  MPSGraphTensorData* dgates_data =
      TensorDataForTensor(dgates_all.get(), TF_FLOAT, device, status);
  if (dgates_data == nil) return;
  [feeds addObject:dgates_data];

  ScopedTensor* outs[5] = {&w_grad, &b_grad, &wci_grad, &wcf_grad, &wco_grad};
  NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
  for (int i = 0; i < 5; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(outs[i]->get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [results addObject:data];
  }
  RunGraph(stream, *reduce, feeds, results, status);
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

void BlockLSTM_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<RnnOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: BlockLSTM has no state.");
  } else {
    BlockLSTM_ComputeImpl(op, ctx, GateLayout(), status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void BlockLSTMV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<RnnOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: BlockLSTMV2 has no state.");
  } else {
    GateLayout layout;
    layout.ifco = true;
    BlockLSTM_ComputeImpl(op, ctx, layout, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void BlockLSTMGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<RnnOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: BlockLSTMGrad has no state.");
  } else {
    BlockLSTMGrad_ComputeImpl(op, ctx, GateLayout(), status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void BlockLSTMGradV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<RnnOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: BlockLSTMGradV2 has no state.");
  } else {
    GateLayout layout;
    layout.ifco = true;
    BlockLSTMGrad_ComputeImpl(op, ctx, layout, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

METAL_RNN_COMPUTE(LSTMCell_Compute, LSTMCell_ComputeImpl)
METAL_RNN_COMPUTE(LSTMCellGrad_Compute, LSTMCellGrad_ComputeImpl)
METAL_RNN_COMPUTE(GRUCell_Compute, GRUCell_ComputeImpl)
METAL_RNN_COMPUTE(GRUCellGrad_Compute, GRUCellGrad_ComputeImpl)

#undef METAL_RNN_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name,
              void* (*create)(TF_OpKernelConstruction*) = &RnnOp_Create,
              const std::vector<const char*>& host_inputs = {}) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, create, compute, &RnnOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
  for (const char* input : host_inputs) {
    TF_KernelBuilder_HostMemory(builder, input);
  }
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

  // seq_len_max bounds the loop on the host, so it is read there.
  Register("BlockLSTM", &BlockLSTM_Compute, "MetalBlockLSTM", &RnnOp_Create,
           {"seq_len_max"});
  Register("BlockLSTMGrad", &BlockLSTMGrad_Compute, "MetalBlockLSTMGrad",
           &RnnOp_Create, {"seq_len_max"});
  Register("BlockLSTMV2", &BlockLSTMV2_Compute, "MetalBlockLSTMV2",
           &RnnOpV2_Create, {"seq_len_max"});
  Register("BlockLSTMGradV2", &BlockLSTMGradV2_Compute,
           "MetalBlockLSTMGradV2", &RnnOpV2_Create, {"seq_len_max"});
}

}  // namespace metal
}  // namespace tensorflow
