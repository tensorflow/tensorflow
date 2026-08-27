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
#include <cstring>
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

// The CudnnRNN family.
//
// These ops are named for cuDNN, and the thing that makes them portable is
// that their parameter buffer is opaque: nothing outside the ops may interpret
// it, and the two canonical conversions are the only defined way in and out.
// So the layout here is this backend's own, and it is used consistently by the
// forward pass, the gradient, and both conversions, which are generated from
// one description so they cannot drift apart.
//
// The recurrence itself is built as one graph unrolled over the sequence, and
// the gradient is that same graph differentiated by MPSGraph. Writing the
// backward pass by hand for four cell types, two directions and any number of
// layers would be a great deal of arithmetic to get subtly wrong; the adjoint
// of a graph is the graph's own business.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

enum class Mode { kLstm, kGru, kRnnTanh, kRnnRelu };

struct RnnSpec {
  Mode mode = Mode::kLstm;
  bool bidirectional = false;
  bool time_major = true;
  bool is_training = true;
  int64_t num_layers = 1;
  int64_t num_units = 1;
  int64_t input_size = 1;
  int64_t batch = 1;
  int64_t seq_length = 1;
  bool has_lengths = false;

  int64_t gates() const {
    switch (mode) {
      case Mode::kLstm:
        return 4;
      case Mode::kGru:
        return 3;
      default:
        return 1;
    }
  }
  int64_t directions() const { return bidirectional ? 2 : 1; }
  // Every layer past the first reads the previous layer's output, which is
  // twice as wide when both directions run.
  int64_t layer_input(int64_t layer) const {
    return layer == 0 ? input_size : num_units * directions();
  }
};

// Where one matrix or bias vector sits inside the opaque buffer.
struct ParamSlot {
  int64_t offset = 0;
  int64_t rows = 0;
  int64_t cols = 0;  // 1 for a bias
  int64_t size() const { return rows * cols; }
};

// The layout, in one place: all weight matrices in layer, direction, kind and
// gate order, then all bias vectors in the same order. Both conversions and
// the graph read this, so there is one description rather than three.
void BuildLayout(const RnnSpec& spec, std::vector<ParamSlot>* weights,
                 std::vector<ParamSlot>* biases) {
  weights->clear();
  biases->clear();
  int64_t offset = 0;
  const int64_t gates = spec.gates();
  for (int64_t layer = 0; layer < spec.num_layers; ++layer) {
    for (int64_t dir = 0; dir < spec.directions(); ++dir) {
      const int64_t in = spec.layer_input(layer);
      for (int64_t g = 0; g < gates; ++g) {
        weights->push_back({offset, spec.num_units, in});
        offset += spec.num_units * in;
      }
      for (int64_t g = 0; g < gates; ++g) {
        weights->push_back({offset, spec.num_units, spec.num_units});
        offset += spec.num_units * spec.num_units;
      }
    }
  }
  for (int64_t layer = 0; layer < spec.num_layers; ++layer) {
    for (int64_t dir = 0; dir < spec.directions(); ++dir) {
      for (int64_t g = 0; g < 2 * gates; ++g) {
        biases->push_back({offset, spec.num_units, 1});
        offset += spec.num_units;
      }
    }
  }
}

int64_t ParamsSize(const RnnSpec& spec) {
  std::vector<ParamSlot> weights, biases;
  BuildLayout(spec, &weights, &biases);
  int64_t total = 0;
  for (const ParamSlot& s : weights) total += s.size();
  for (const ParamSlot& s : biases) total += s.size();
  return total;
}

struct RnnOp {
  RnnSpec spec;
  TF_DataType dtype = TF_FLOAT;
  TF_DataType size_dtype = TF_INT32;
  int32_t num_params = 0;
  bool valid = false;
};

bool ReadMode(TF_OpKernelConstruction* ctx, RnnSpec* spec,
              TF_Status* status) {
  char text[32] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "rnn_mode", text,
                                        sizeof(text) - 1, status);
  if (TF_GetCode(status) == TF_OK && text[0] != '\0') {
    if (std::strcmp(text, "lstm") == 0) {
      spec->mode = Mode::kLstm;
    } else if (std::strcmp(text, "gru") == 0) {
      spec->mode = Mode::kGru;
    } else if (std::strcmp(text, "rnn_tanh") == 0) {
      spec->mode = Mode::kRnnTanh;
    } else if (std::strcmp(text, "rnn_relu") == 0) {
      spec->mode = Mode::kRnnRelu;
    } else {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: unknown recurrent mode.");
      return false;
    }
  }
  TF_SetStatus(status, TF_OK, "");

  std::memset(text, 0, sizeof(text));
  TF_OpKernelConstruction_GetAttrString(ctx, "direction", text,
                                        sizeof(text) - 1, status);
  if (TF_GetCode(status) == TF_OK && text[0] != '\0') {
    spec->bidirectional = std::strcmp(text, "bidirectional") == 0;
  }
  TF_SetStatus(status, TF_OK, "");

  std::memset(text, 0, sizeof(text));
  TF_OpKernelConstruction_GetAttrString(ctx, "input_mode", text,
                                        sizeof(text) - 1, status);
  if (TF_GetCode(status) == TF_OK && text[0] != '\0' &&
      std::strcmp(text, "linear_input") != 0 &&
      std::strcmp(text, "auto_select") != 0) {
    // skip_input drops the input matrix entirely, which changes the layout
    // and only works when the widths already agree.
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: the recurrent ops implement linear_input only.");
    return false;
  }
  TF_SetStatus(status, TF_OK, "");

  float dropout = 0.0f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "dropout", &dropout, status);
  if (TF_GetCode(status) == TF_OK && dropout != 0.0f) {
    // Dropout between layers would have to reproduce cuDNN's own generator to
    // be reproducible, and nothing defines that sequence outside cuDNN.
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: the recurrent ops implement dropout of zero only.");
    return false;
  }
  TF_SetStatus(status, TF_OK, "");

  int32_t num_proj = 0;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_proj", &num_proj, status);
  if (TF_GetCode(status) == TF_OK && num_proj != 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: the recurrent ops implement a projection size of "
                 "zero only.");
    return false;
  }
  TF_SetStatus(status, TF_OK, "");

  TF_Bool flag = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "is_training", &flag, status);
  if (TF_GetCode(status) == TF_OK) spec->is_training = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  flag = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "time_major", &flag, status);
  if (TF_GetCode(status) == TF_OK) spec->time_major = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  return true;
}

void* RnnOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new RnnOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  TF_OpKernelConstruction_GetAttrType(ctx, "S", &op->size_dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->size_dtype = TF_INT32;
  }
  int32_t num_params = 0;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_params", &num_params, status);
  if (TF_GetCode(status) == TF_OK) op->num_params = num_params;
  TF_SetStatus(status, TF_OK, "");
  if (!ReadMode(ctx, &op->spec, status)) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  op->valid = true;
  TF_DeleteStatus(status);
  return op;
}

void RnnOp_Delete(void* kernel) { delete static_cast<RnnOp*>(kernel); }

// Reads one of the three size scalars, which arrive in host memory.
bool ReadScalarInt(TF_OpKernelContext* ctx, int index, int64_t* out,
                   TF_Status* status) {
  ScopedTensor t;
  TF_GetInput(ctx, index, t.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  const void* data = TF_TensorData(t.get());
  if (data == nullptr || TF_TensorElementCount(t.get()) < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a recurrent size argument has no data.");
    return false;
  }
  *out = TF_TensorType(t.get()) == TF_INT64
             ? *static_cast<const int64_t*>(data)
             : *static_cast<const int32_t*>(data);
  return true;
}

MPSGraphTensor* Matrix(MPSGraph* g, MPSGraphTensor* params,
                       const ParamSlot& slot) {
  MPSGraphTensor* flat =
      [g sliceTensor:params
           dimension:0
               start:static_cast<NSInteger>(slot.offset)
              length:static_cast<NSInteger>(slot.size())
                name:nil];
  return [g reshapeTensor:flat
                withShape:@[
                  @(static_cast<NSInteger>(slot.rows)),
                  @(static_cast<NSInteger>(slot.cols))
                ]
                     name:nil];
}

MPSGraphTensor* Row(MPSGraph* g, MPSGraphTensor* params,
                    const ParamSlot& slot) {
  MPSGraphTensor* flat =
      [g sliceTensor:params
           dimension:0
               start:static_cast<NSInteger>(slot.offset)
              length:static_cast<NSInteger>(slot.size())
                name:nil];
  return [g reshapeTensor:flat
                withShape:@[ @1, @(static_cast<NSInteger>(slot.rows)) ]
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
MPSGraphTensor* MatMulT(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* w) {
  // The canonical weights are stored as [units, in], and the data is
  // [batch, in], so the multiply transposes the weights.
  return [g matrixMultiplicationWithPrimaryTensor:x
                                  secondaryTensor:[g transposeTensor:w
                                                           dimension:0
                                                       withDimension:1
                                                                name:nil]
                                             name:nil];
}

// What the whole stack computes, as one graph. `outputs` receives the
// sequence output, the final hidden state and the final cell state.
struct RnnGraph {
  MPSGraphTensor* output = nil;
  MPSGraphTensor* output_h = nil;
  MPSGraphTensor* output_c = nil;
};

RnnGraph BuildStack(MPSGraph* g, const RnnSpec& spec, MPSDataType dtype,
                    MPSGraphTensor* input, MPSGraphTensor* input_h,
                    MPSGraphTensor* input_c, MPSGraphTensor* params,
                    MPSGraphTensor* lengths) {
  std::vector<ParamSlot> weights, biases;
  BuildLayout(spec, &weights, &biases);
  const int64_t gates = spec.gates();
  const int64_t dirs = spec.directions();
  const int64_t T = spec.seq_length;
  const int64_t B = spec.batch;
  const int64_t U = spec.num_units;

  MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dtype];

  // The step input, as a list over time of [batch, width].
  std::vector<MPSGraphTensor*> layer_input;
  layer_input.reserve(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    MPSGraphTensor* step = [g sliceTensor:input
                                dimension:0
                                    start:static_cast<NSInteger>(t)
                                   length:1
                                     name:nil];
    layer_input.push_back([g
        reshapeTensor:step
            withShape:@[
              @(static_cast<NSInteger>(B)),
              @(static_cast<NSInteger>(spec.input_size))
            ]
                 name:nil]);
  }

  std::vector<MPSGraphTensor*> final_h, final_c;
  int64_t weight_index = 0;
  int64_t bias_index = 0;

  for (int64_t layer = 0; layer < spec.num_layers; ++layer) {
    std::vector<std::vector<MPSGraphTensor*>> per_direction(
        static_cast<size_t>(dirs));
    for (int64_t dir = 0; dir < dirs; ++dir) {
      const int64_t state_row = layer * dirs + dir;
      MPSGraphTensor* h = [g sliceTensor:input_h
                               dimension:0
                                   start:static_cast<NSInteger>(state_row)
                                  length:1
                                    name:nil];
      h = [g reshapeTensor:h
                 withShape:@[
                   @(static_cast<NSInteger>(B)), @(static_cast<NSInteger>(U))
                 ]
                      name:nil];
      MPSGraphTensor* c = nil;
      if (spec.mode == Mode::kLstm) {
        c = [g sliceTensor:input_c
                 dimension:0
                     start:static_cast<NSInteger>(state_row)
                    length:1
                      name:nil];
        c = [g reshapeTensor:c
                   withShape:@[
                     @(static_cast<NSInteger>(B)), @(static_cast<NSInteger>(U))
                   ]
                        name:nil];
      }

      // This direction's matrices and biases, in the order the layout lays
      // them down.
      std::vector<MPSGraphTensor*> wx(static_cast<size_t>(gates));
      std::vector<MPSGraphTensor*> wh(static_cast<size_t>(gates));
      std::vector<MPSGraphTensor*> bx(static_cast<size_t>(gates));
      std::vector<MPSGraphTensor*> bh(static_cast<size_t>(gates));
      for (int64_t k = 0; k < gates; ++k) {
        wx[static_cast<size_t>(k)] =
            Matrix(g, params, weights[static_cast<size_t>(weight_index++)]);
      }
      for (int64_t k = 0; k < gates; ++k) {
        wh[static_cast<size_t>(k)] =
            Matrix(g, params, weights[static_cast<size_t>(weight_index++)]);
      }
      for (int64_t k = 0; k < gates; ++k) {
        bx[static_cast<size_t>(k)] =
            Row(g, params, biases[static_cast<size_t>(bias_index++)]);
      }
      for (int64_t k = 0; k < gates; ++k) {
        bh[static_cast<size_t>(k)] =
            Row(g, params, biases[static_cast<size_t>(bias_index++)]);
      }

      std::vector<MPSGraphTensor*> steps(static_cast<size_t>(T), nil);
      for (int64_t step = 0; step < T; ++step) {
        // The reverse direction walks the sequence backwards; everything else
        // about it is the same.
        const int64_t t = dir == 0 ? step : T - 1 - step;
        MPSGraphTensor* x = layer_input[static_cast<size_t>(t)];
        MPSGraphTensor* h_new = nil;
        MPSGraphTensor* c_new = nil;

        if (spec.mode == Mode::kLstm) {
          MPSGraphTensor* gate[4];
          for (int k = 0; k < 4; ++k) {
            gate[k] = Add(g,
                          Add(g, MatMulT(g, x, wx[k]), bx[k]),
                          Add(g, MatMulT(g, h, wh[k]), bh[k]));
          }
          // The canonical order is input, forget, cell, output.
          MPSGraphTensor* i = [g sigmoidWithTensor:gate[0] name:nil];
          MPSGraphTensor* f = [g sigmoidWithTensor:gate[1] name:nil];
          MPSGraphTensor* cell = [g tanhWithTensor:gate[2] name:nil];
          MPSGraphTensor* o = [g sigmoidWithTensor:gate[3] name:nil];
          c_new = Add(g, Mul(g, f, c), Mul(g, i, cell));
          h_new = Mul(g, o, [g tanhWithTensor:c_new name:nil]);
        } else if (spec.mode == Mode::kGru) {
          // The reset gate is applied after the recurrent multiply, which is
          // what distinguishes this from the GRU in GRUBlockCell.
          MPSGraphTensor* r = [g
              sigmoidWithTensor:Add(g, Add(g, MatMulT(g, x, wx[0]), bx[0]),
                                    Add(g, MatMulT(g, h, wh[0]), bh[0]))
                           name:nil];
          MPSGraphTensor* u = [g
              sigmoidWithTensor:Add(g, Add(g, MatMulT(g, x, wx[1]), bx[1]),
                                    Add(g, MatMulT(g, h, wh[1]), bh[1]))
                           name:nil];
          MPSGraphTensor* n = [g
              tanhWithTensor:Add(g, Add(g, MatMulT(g, x, wx[2]), bx[2]),
                                 Mul(g, r, Add(g, MatMulT(g, h, wh[2]),
                                               bh[2])))
                        name:nil];
          h_new = Add(g, Mul(g, Sub(g, [g constantWithScalar:1.0
                                                    dataType:dtype],
                                    u),
                             n),
                      Mul(g, u, h));
        } else {
          MPSGraphTensor* pre =
              Add(g, Add(g, MatMulT(g, x, wx[0]), bx[0]),
                  Add(g, MatMulT(g, h, wh[0]), bh[0]));
          h_new = spec.mode == Mode::kRnnTanh
                      ? [g tanhWithTensor:pre name:nil]
                      : [g maximumWithPrimaryTensor:pre
                                    secondaryTensor:zero
                                               name:nil];
        }

        // Past a sequence's own length the state stops moving and the output
        // is zero, which is what padding a batch of unequal lengths means.
        MPSGraphTensor* emitted = h_new;
        if (lengths != nil) {
          MPSGraphTensor* limit =
              [g constantWithScalar:static_cast<double>(t) dataType:dtype];
          // The mask is arithmetic rather than a select because the two have
          // different derivatives here: a select hands the incoming gradient
          // to both of its branches, so a step that a sequence never took
          // would still collect one. Multiplying by a zero or a one keeps the
          // gradient where the value came from.
          MPSGraphTensor* mask =
              [g castTensor:[g greaterThanWithPrimaryTensor:lengths
                                            secondaryTensor:limit
                                                       name:nil]
                     toType:dtype
                       name:nil];
          h_new = Add(g, h, Mul(g, mask, Sub(g, h_new, h)));
          if (c_new != nil) {
            c_new = Add(g, c, Mul(g, mask, Sub(g, c_new, c)));
          }
          emitted = Mul(g, mask, h_new);
        }
        h = h_new;
        if (c_new != nil) c = c_new;
        steps[static_cast<size_t>(t)] = emitted;
      }
      per_direction[static_cast<size_t>(dir)] = steps;
      final_h.push_back(h);
      if (spec.mode == Mode::kLstm) final_c.push_back(c);
    }

    // The next layer reads this one, with both directions side by side.
    for (int64_t t = 0; t < T; ++t) {
      MPSGraphTensor* joined = per_direction[0][static_cast<size_t>(t)];
      if (dirs == 2) {
        joined = [g concatTensor:joined
                      withTensor:per_direction[1][static_cast<size_t>(t)]
                       dimension:1
                            name:nil];
      }
      layer_input[static_cast<size_t>(t)] = joined;
    }
  }

  // The sequence output, stacked back into [time, batch, directions * units].
  NSMutableArray<MPSGraphTensor*>* stacked = [NSMutableArray array];
  for (int64_t t = 0; t < T; ++t) {
    [stacked addObject:[g reshapeTensor:layer_input[static_cast<size_t>(t)]
                              withShape:@[
                                @1, @(static_cast<NSInteger>(B)),
                                @(static_cast<NSInteger>(U * dirs))
                              ]
                                   name:nil]];
  }
  RnnGraph out;
  out.output = [g concatTensors:stacked dimension:0 name:nil];

  NSMutableArray<MPSGraphTensor*>* states = [NSMutableArray array];
  for (MPSGraphTensor* h : final_h) {
    [states addObject:[g reshapeTensor:h
                             withShape:@[
                               @1, @(static_cast<NSInteger>(B)),
                               @(static_cast<NSInteger>(U))
                             ]
                                  name:nil]];
  }
  out.output_h = [g concatTensors:states dimension:0 name:nil];
  if (spec.mode == Mode::kLstm) {
    NSMutableArray<MPSGraphTensor*>* cells = [NSMutableArray array];
    for (MPSGraphTensor* c : final_c) {
      [cells addObject:[g reshapeTensor:c
                              withShape:@[
                                @1, @(static_cast<NSInteger>(B)),
                                @(static_cast<NSInteger>(U))
                              ]
                                   name:nil]];
    }
    out.output_c = [g concatTensors:cells dimension:0 name:nil];
  }
  return out;
}

void AppendSpecToKey(const RnnSpec& spec, std::string* key) {
  key->append("/m").append(std::to_string(static_cast<int>(spec.mode)));
  key->append(spec.bidirectional ? "/bi" : "/uni");
  key->append("/l").append(std::to_string(spec.num_layers));
  key->append("/u").append(std::to_string(spec.num_units));
  key->append("/i").append(std::to_string(spec.input_size));
  key->append("/b").append(std::to_string(spec.batch));
  key->append("/t").append(std::to_string(spec.seq_length));
  key->append(spec.has_lengths ? "/masked" : "/full");
}

}  // namespace

// Defined below the anonymous namespace so the registrations can see them.
void RegisterMetalCudnnRnnKernels();

namespace {

/*** PARAMS SIZE ***/

void ParamsSize_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  RnnSpec spec = op->spec;
  if (!ReadScalarInt(ctx, 0, &spec.num_layers, status)) return;
  if (!ReadScalarInt(ctx, 1, &spec.num_units, status)) return;
  if (!ReadScalarInt(ctx, 2, &spec.input_size, status)) return;
  const int64_t total = ParamsSize(spec);

  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, op->size_dtype, nullptr, 0,
                                 TF_DataTypeSize(op->size_dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  // A freshly allocated scalar with nothing in flight against it, so the host
  // may write it directly; that is what unified memory is for.
  void* data = TF_TensorData(output.get());
  if (data == nullptr) return;
  if (op->size_dtype == TF_INT64) {
    *static_cast<int64_t*>(data) = total;
  } else {
    *static_cast<int32_t*>(data) = static_cast<int32_t>(total);
  }
}

/*** THE TWO CANONICAL CONVERSIONS ***/

// Both are pure movement between one flat buffer and a list of matrices, so
// they are blits at the offsets the layout gives.
void Canonical_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx, bool to_params,
                           TF_Status* status) {
  RnnSpec spec = op->spec;
  if (!ReadScalarInt(ctx, 0, &spec.num_layers, status)) return;
  if (!ReadScalarInt(ctx, 1, &spec.num_units, status)) return;
  if (!ReadScalarInt(ctx, 2, &spec.input_size, status)) return;

  std::vector<ParamSlot> weights, biases;
  BuildLayout(spec, &weights, &biases);
  const int64_t expected = static_cast<int64_t>(weights.size());
  if (op->num_params != expected) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: num_params does not match this configuration.");
    return;
  }
  const int64_t total = ParamsSize(spec);
  const size_t element = TF_DataTypeSize(op->dtype);

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  ScopedTensor params;
  std::vector<ScopedTensor> canonical(static_cast<size_t>(2 * expected));
  if (to_params) {
    const std::vector<int64_t> shape = {total};
    params.reset(TF_AllocateOutput(ctx, 0, op->dtype, shape.data(), 1,
                                   static_cast<size_t>(total) * element,
                                   status));
    if (TF_GetCode(status) != TF_OK) return;
    for (int64_t i = 0; i < 2 * expected; ++i) {
      TF_GetInput(ctx, static_cast<int>(3 + i),
                  canonical[static_cast<size_t>(i)].address(), status);
      if (TF_GetCode(status) != TF_OK) return;
    }
  } else {
    TF_GetInput(ctx, 3, params.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    for (int64_t i = 0; i < expected; ++i) {
      const ParamSlot& slot = weights[static_cast<size_t>(i)];
      const std::vector<int64_t> shape = {slot.rows, slot.cols};
      canonical[static_cast<size_t>(i)].reset(TF_AllocateOutput(
          ctx, static_cast<int>(i), op->dtype, shape.data(), 2,
          static_cast<size_t>(slot.size()) * element, status));
      if (TF_GetCode(status) != TF_OK) return;
    }
    for (int64_t i = 0; i < expected; ++i) {
      const ParamSlot& slot = biases[static_cast<size_t>(i)];
      const std::vector<int64_t> shape = {slot.rows};
      canonical[static_cast<size_t>(expected + i)].reset(TF_AllocateOutput(
          ctx, static_cast<int>(expected + i), op->dtype, shape.data(), 1,
          static_cast<size_t>(slot.size()) * element, status));
      if (TF_GetCode(status) != TF_OK) return;
    }
  }

  BufferSlice params_slice;
  if (!SliceForTensor(params.get(), &params_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a parameter "
                 "conversion.");
    return;
  }
  id<MTLBlitCommandEncoder> encoder =
      [command_buffer.get() blitCommandEncoder];
  for (int64_t i = 0; i < 2 * expected; ++i) {
    const ParamSlot& slot = i < expected
                                ? weights[static_cast<size_t>(i)]
                                : biases[static_cast<size_t>(i - expected)];
    BufferSlice piece;
    if (!SliceForTensor(canonical[static_cast<size_t>(i)].get(), &piece,
                        status)) {
      return;
    }
    const NSUInteger bytes = static_cast<NSUInteger>(slot.size()) * element;
    const size_t at = params_slice.offset +
                      static_cast<size_t>(slot.offset) * element;
    if (to_params) {
      [encoder copyFromBuffer:piece.buffer
                 sourceOffset:piece.offset
                     toBuffer:params_slice.buffer
            destinationOffset:at
                         size:bytes];
    } else {
      [encoder copyFromBuffer:params_slice.buffer
                 sourceOffset:at
                     toBuffer:piece.buffer
            destinationOffset:piece.offset
                         size:bytes];
    }
  }
  [encoder endEncoding];
  command_buffer.Commit();
}

/*** FORWARD ***/

// `lengths_index` is where the sequence lengths sit, or -1 when the op has
// none; `extra_outputs` is how many trailing outputs exist beyond the four.
void Forward_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx,
                         int lengths_index, TF_Status* status) {
  ScopedTensor input, input_h, input_c, params, lengths;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, input_h.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, input_c.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, params.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (lengths_index >= 0) {
    TF_GetInput(ctx, lengths_index, lengths.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  RnnSpec spec = op->spec;
  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> h_shape = ShapeOf(input_h.get());
  if (in_shape.size() != 3 || h_shape.size() != 3) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the recurrent ops expect rank-3 inputs.");
    return;
  }
  // Time major puts the sequence first; otherwise the batch does.
  spec.seq_length = spec.time_major ? in_shape[0] : in_shape[1];
  spec.batch = spec.time_major ? in_shape[1] : in_shape[0];
  spec.input_size = in_shape[2];
  spec.num_units = h_shape[2];
  spec.num_layers = h_shape[0] / spec.directions();
  spec.has_lengths = lengths_index >= 0;

  const int64_t dirs = spec.directions();
  std::vector<int64_t> out_shape = in_shape;
  if (spec.time_major) {
    out_shape = {spec.seq_length, spec.batch, spec.num_units * dirs};
  } else {
    out_shape = {spec.batch, spec.seq_length, spec.num_units * dirs};
  }
  const std::vector<int64_t> state_shape = {h_shape[0], spec.batch,
                                            spec.num_units};

  const size_t element = TF_DataTypeSize(op->dtype);
  ScopedTensor output, output_h, output_c, reserve;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), 3,
      static_cast<size_t>(ElementCount(out_shape)) * element, status));
  if (TF_GetCode(status) != TF_OK) return;
  output_h.reset(TF_AllocateOutput(
      ctx, 1, op->dtype, state_shape.data(), 3,
      static_cast<size_t>(ElementCount(state_shape)) * element, status));
  if (TF_GetCode(status) != TF_OK) return;
  output_c.reset(TF_AllocateOutput(
      ctx, 2, op->dtype, state_shape.data(), 3,
      static_cast<size_t>(ElementCount(state_shape)) * element, status));
  if (TF_GetCode(status) != TF_OK) return;
  // The reserve space is opaque and this backend keeps nothing in it: the
  // gradient rebuilds the forward pass from the inputs it is given, which
  // costs time and saves having to define a second layout.
  const std::vector<int64_t> reserve_shape = {0};
  reserve.reset(TF_AllocateOutput(ctx, 3, op->dtype, reserve_shape.data(), 1,
                                  0, status));
  if (TF_GetCode(status) != TF_OK) return;
  for (int i = 4; i < TF_NumOutputs(ctx); ++i) {
    ScopedTensor extra;
    const std::vector<int64_t> empty = {0};
    extra.reset(TF_AllocateOutput(ctx, i, TF_INT8, empty.data(), 1, 0,
                                  status));
    if (TF_GetCode(status) != TF_OK) return;
  }
  if (ElementCount(out_shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  const int64_t total_params = ParamsSize(spec);
  const std::vector<int64_t> params_shape = {total_params};
  const std::vector<int64_t> graph_in_shape = {spec.seq_length, spec.batch,
                                               spec.input_size};
  const std::vector<int64_t> lengths_shape = {spec.batch};

  std::string key = "CudnnRNNForward";
  AppendSpecToKey(spec, &key);
  key.append(spec.time_major ? "/tm" : "/bm");
  key.append("/d").append(std::to_string(static_cast<int>(op->dtype)));

  const RnnSpec captured = spec;
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* h0 = [g placeholderWithShape:MPSShape(state_shape)
                                            dataType:mps_dtype
                                                name:nil];
        MPSGraphTensor* c0 = [g placeholderWithShape:MPSShape(state_shape)
                                            dataType:mps_dtype
                                                name:nil];
        MPSGraphTensor* p = [g placeholderWithShape:MPSShape(params_shape)
                                           dataType:mps_dtype
                                               name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:h0];
        [out->inputs addObject:c0];
        [out->inputs addObject:p];
        MPSGraphTensor* len = nil;
        if (captured.has_lengths) {
          len = [g placeholderWithShape:MPSShape(lengths_shape)
                               dataType:MPSDataTypeInt32
                                   name:nil];
          [out->inputs addObject:len];
          // Compared against a time index, so it joins the data's own type.
          len = [g castTensor:len toType:mps_dtype name:nil];
          len = [g reshapeTensor:len
                       withShape:@[ @(static_cast<NSInteger>(captured.batch)),
                                    @1 ]
                            name:nil];
        }
        // The recurrence is written time first; a batch-first caller is
        // transposed on the way in and back on the way out.
        MPSGraphTensor* sequence =
            captured.time_major
                ? x
                : [g transposeTensor:x dimension:0 withDimension:1 name:nil];
        RnnGraph built = BuildStack(g, captured, mps_dtype, sequence, h0, c0,
                                    p, len);
        MPSGraphTensor* result =
            captured.time_major
                ? built.output
                : [g transposeTensor:built.output
                           dimension:0
                       withDimension:1
                                name:nil];
        [out->outputs addObject:result];
        [out->outputs addObject:built.output_h];
        [out->outputs addObject:built.output_c != nil
                                    ? built.output_c
                                    : [g constantWithScalar:0.0
                                                      shape:MPSShape(
                                                                state_shape)
                                                   dataType:mps_dtype]];
      },
      status);
  if (cached == nullptr) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* h_data =
      TensorDataForTensor(input_h.get(), op->dtype, device, status);
  if (h_data == nil) return;
  MPSGraphTensorData* c_data =
      TensorDataForTensor(input_c.get(), op->dtype, device, status);
  if (c_data == nil) return;
  MPSGraphTensorData* p_data =
      TensorDataForTensor(params.get(), op->dtype, device, status);
  if (p_data == nil) return;
  [feeds addObject:x_data];
  [feeds addObject:h_data];
  [feeds addObject:c_data];
  [feeds addObject:p_data];
  if (spec.has_lengths) {
    MPSGraphTensorData* len_data =
        TensorDataForTensor(lengths.get(), TF_INT32, device, status);
    if (len_data == nil) return;
    [feeds addObject:len_data];
  }

  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  MPSGraphTensorData* oh_data =
      TensorDataForTensor(output_h.get(), op->dtype, device, status);
  if (oh_data == nil) return;
  MPSGraphTensorData* oc_data =
      TensorDataForTensor(output_c.get(), op->dtype, device, status);
  if (oc_data == nil) return;
  RunGraph(stream, *cached, feeds, @[ out_data, oh_data, oc_data ], status);
}

/*** GRADIENT ***/

void Backward_ComputeImpl(RnnOp* op, TF_OpKernelContext* ctx,
                          int lengths_index, TF_Status* status) {
  ScopedTensor in[11];
  for (int i = 0; i < 11; ++i) {
    TF_GetInput(ctx, i, in[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  ScopedTensor lengths;
  if (lengths_index >= 0) {
    TF_GetInput(ctx, lengths_index, lengths.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }
  ScopedTensor& input = in[0];
  ScopedTensor& input_h = in[1];
  ScopedTensor& input_c = in[2];
  ScopedTensor& params = in[3];
  ScopedTensor& output_backprop = in[7];
  ScopedTensor& output_h_backprop = in[8];
  ScopedTensor& output_c_backprop = in[9];

  RnnSpec spec = op->spec;
  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> h_shape = ShapeOf(input_h.get());
  if (in_shape.size() != 3 || h_shape.size() != 3) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the recurrent gradient expects rank-3 inputs.");
    return;
  }
  spec.seq_length = spec.time_major ? in_shape[0] : in_shape[1];
  spec.batch = spec.time_major ? in_shape[1] : in_shape[0];
  spec.input_size = in_shape[2];
  spec.num_units = h_shape[2];
  spec.num_layers = h_shape[0] / spec.directions();
  spec.has_lengths = lengths_index >= 0;

  const std::vector<int64_t> state_shape = {h_shape[0], spec.batch,
                                            spec.num_units};
  const int64_t total_params = ParamsSize(spec);
  const std::vector<int64_t> params_shape = {total_params};
  const std::vector<int64_t> out_shape = ShapeOf(output_backprop.get());
  const std::vector<int64_t> lengths_shape = {spec.batch};
  const size_t element = TF_DataTypeSize(op->dtype);

  ScopedTensor dx, dh, dc, dparams;
  dx.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, in_shape.data(), 3,
      static_cast<size_t>(ElementCount(in_shape)) * element, status));
  if (TF_GetCode(status) != TF_OK) return;
  dh.reset(TF_AllocateOutput(
      ctx, 1, op->dtype, state_shape.data(), 3,
      static_cast<size_t>(ElementCount(state_shape)) * element, status));
  if (TF_GetCode(status) != TF_OK) return;
  dc.reset(TF_AllocateOutput(
      ctx, 2, op->dtype, state_shape.data(), 3,
      static_cast<size_t>(ElementCount(state_shape)) * element, status));
  if (TF_GetCode(status) != TF_OK) return;
  dparams.reset(TF_AllocateOutput(
      ctx, 3, op->dtype, params_shape.data(), 1,
      static_cast<size_t>(total_params) * element, status));
  if (TF_GetCode(status) != TF_OK) return;
  if (ElementCount(in_shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "CudnnRNNBackward";
  AppendSpecToKey(spec, &key);
  key.append(spec.time_major ? "/tm" : "/bm");
  key.append("/d").append(std::to_string(static_cast<int>(op->dtype)));

  const RnnSpec captured = spec;
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* h0 = [g placeholderWithShape:MPSShape(state_shape)
                                            dataType:mps_dtype
                                                name:nil];
        MPSGraphTensor* c0 = [g placeholderWithShape:MPSShape(state_shape)
                                            dataType:mps_dtype
                                                name:nil];
        MPSGraphTensor* p = [g placeholderWithShape:MPSShape(params_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* dy = [g placeholderWithShape:MPSShape(out_shape)
                                            dataType:mps_dtype
                                                name:nil];
        MPSGraphTensor* dhy = [g placeholderWithShape:MPSShape(state_shape)
                                             dataType:mps_dtype
                                                 name:nil];
        MPSGraphTensor* dcy = [g placeholderWithShape:MPSShape(state_shape)
                                             dataType:mps_dtype
                                                 name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:h0];
        [out->inputs addObject:c0];
        [out->inputs addObject:p];
        [out->inputs addObject:dy];
        [out->inputs addObject:dhy];
        [out->inputs addObject:dcy];
        MPSGraphTensor* len = nil;
        if (captured.has_lengths) {
          len = [g placeholderWithShape:MPSShape(lengths_shape)
                               dataType:MPSDataTypeInt32
                                   name:nil];
          [out->inputs addObject:len];
          len = [g castTensor:len toType:mps_dtype name:nil];
          len = [g reshapeTensor:len
                       withShape:@[ @(static_cast<NSInteger>(captured.batch)),
                                    @1 ]
                            name:nil];
        }

        MPSGraphTensor* sequence =
            captured.time_major
                ? x
                : [g transposeTensor:x dimension:0 withDimension:1 name:nil];
        RnnGraph built = BuildStack(g, captured, mps_dtype, sequence, h0, c0,
                                    p, len);
        MPSGraphTensor* result =
            captured.time_major
                ? built.output
                : [g transposeTensor:built.output
                           dimension:0
                       withDimension:1
                                name:nil];

        // The adjoint of the forward pass is the derivative of the sum of its
        // outputs weighted by the gradients arriving at them. Differentiating
        // that one scalar gives every gradient the op has to return, without a
        // line of backward arithmetic written by hand.
        MPSGraphTensor* loss = [g
            reductionSumWithTensor:Mul(g, result, dy)
                              axes:nil
                              name:nil];
        loss = Add(g, loss,
                   [g reductionSumWithTensor:Mul(g, built.output_h, dhy)
                                        axes:nil
                                        name:nil]);
        if (built.output_c != nil) {
          loss = Add(g, loss,
                     [g reductionSumWithTensor:Mul(g, built.output_c, dcy)
                                          axes:nil
                                          name:nil]);
        }
        // Asking for the derivative with respect to something the loss does
        // not depend on is a hard error rather than a zero, so only the
        // tensors the cell actually reads are asked for. Everything else is
        // zero, which is the right answer anyway: the initial cell state of a
        // network that has no cell state cannot change the output.
        NSMutableArray<MPSGraphTensor*>* wanted =
            [NSMutableArray arrayWithObjects:x, h0, p, nil];
        if (built.output_c != nil) [wanted insertObject:c0 atIndex:2];
        NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
            [g gradientForPrimaryTensor:loss withTensors:wanted name:nil];
        MPSGraphTensor* zero_state =
            [g constantWithScalar:0.0
                            shape:MPSShape(state_shape)
                         dataType:mps_dtype];
        MPSGraphTensor* zero_params =
            [g constantWithScalar:0.0
                            shape:MPSShape(params_shape)
                         dataType:mps_dtype];
        MPSGraphTensor* zero_input =
            [g constantWithScalar:0.0
                            shape:MPSShape(in_shape)
                         dataType:mps_dtype];
        // A state the cell never reads, the initial cell state of a plain
        // recurrent network among them, simply has no gradient.
        [out->outputs addObject:grads[x] != nil ? grads[x] : zero_input];
        [out->outputs addObject:grads[h0] != nil ? grads[h0] : zero_state];
        [out->outputs addObject:grads[c0] != nil ? grads[c0] : zero_state];
        [out->outputs addObject:grads[p] != nil ? grads[p] : zero_params];
      },
      status);
  if (cached == nullptr) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  ScopedTensor* fed[7] = {&input,  &input_h,          &input_c,
                          &params, &output_backprop,  &output_h_backprop,
                          &output_c_backprop};
  for (int i = 0; i < 7; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(fed[i]->get(), op->dtype, device, status);
    if (data == nil) return;
    [feeds addObject:data];
  }
  if (spec.has_lengths) {
    MPSGraphTensorData* len_data =
        TensorDataForTensor(lengths.get(), TF_INT32, device, status);
    if (len_data == nil) return;
    [feeds addObject:len_data];
  }

  NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
  ScopedTensor* produced[4] = {&dx, &dh, &dc, &dparams};
  for (int i = 0; i < 4; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(produced[i]->get(), op->dtype, device, status);
    if (data == nil) return;
    [results addObject:data];
  }
  RunGraph(stream, *cached, feeds, results, status);
}

#define METAL_RNN_COMPUTE(NAME, BODY)                                       \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<RnnOp*>(kernel);                                 \
    if (op == nullptr || !op->valid) {                                      \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a recurrent kernel has no state.");              \
    } else {                                                                \
      BODY;                                                                 \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_RNN_COMPUTE(ParamsSize_Compute, ParamsSize_ComputeImpl(op, ctx, status))
METAL_RNN_COMPUTE(ToParams_Compute,
                  Canonical_ComputeImpl(op, ctx, /*to_params=*/true, status))
METAL_RNN_COMPUTE(ToCanonical_Compute,
                  Canonical_ComputeImpl(op, ctx, /*to_params=*/false, status))
METAL_RNN_COMPUTE(Forward_Compute, Forward_ComputeImpl(op, ctx, -1, status))
METAL_RNN_COMPUTE(ForwardV3_Compute, Forward_ComputeImpl(op, ctx, 4, status))
METAL_RNN_COMPUTE(Backward_Compute, Backward_ComputeImpl(op, ctx, -1, status))
METAL_RNN_COMPUTE(BackwardV3_Compute, Backward_ComputeImpl(op, ctx, 12, status))

#undef METAL_RNN_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &RnnOp_Create, compute, &RnnOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalCudnnRnnKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  // The three size arguments describe the shape of the work, so they are read
  // on the host, as they are in the kernels this replaces.
  static const std::vector<const char*> kSizes = {"num_layers", "num_units",
                                                  "input_size"};
  for (int i = 0; i < 2; ++i) {
    const std::string suffix = kSuffixes[i];
    Register("CudnnRNNParamsSize", &ParamsSize_Compute, kDTypes[i],
             "MetalCudnnRNNParamsSize" + suffix, kSizes);
    Register("CudnnRNNCanonicalToParams", &ToParams_Compute, kDTypes[i],
             "MetalCudnnRNNCanonicalToParams" + suffix, kSizes);
    Register("CudnnRNNCanonicalToParamsV2", &ToParams_Compute, kDTypes[i],
             "MetalCudnnRNNCanonicalToParamsV2" + suffix, kSizes);
    Register("CudnnRNNParamsToCanonical", &ToCanonical_Compute, kDTypes[i],
             "MetalCudnnRNNParamsToCanonical" + suffix, kSizes);
    Register("CudnnRNNParamsToCanonicalV2", &ToCanonical_Compute, kDTypes[i],
             "MetalCudnnRNNParamsToCanonicalV2" + suffix, kSizes);
    Register("CudnnRNN", &Forward_Compute, kDTypes[i],
             "MetalCudnnRNN" + suffix, {});
    Register("CudnnRNNV2", &Forward_Compute, kDTypes[i],
             "MetalCudnnRNNV2" + suffix, {});
    Register("CudnnRNNV3", &ForwardV3_Compute, kDTypes[i],
             "MetalCudnnRNNV3" + suffix, {"sequence_lengths"});
    Register("CudnnRNNBackprop", &Backward_Compute, kDTypes[i],
             "MetalCudnnRNNBackprop" + suffix, {});
    Register("CudnnRNNBackpropV2", &Backward_Compute, kDTypes[i],
             "MetalCudnnRNNBackpropV2" + suffix, {});
    Register("CudnnRNNBackpropV3", &BackwardV3_Compute, kDTypes[i],
             "MetalCudnnRNNBackpropV3" + suffix, {"sequence_lengths"});
  }
}

}  // namespace metal
}  // namespace tensorflow
