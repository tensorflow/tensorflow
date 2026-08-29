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

#include "tensorflow/core/common_runtime/metal/metal_graph.h"

#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"

namespace tensorflow {
namespace metal {
namespace {

// This file reads and writes GraphDef on the wire rather than through
// TensorFlow's generated C++ classes, for the same reason the profiler encodes
// XSpace by hand: those classes live in libtensorflow_framework's C++ ABI, and
// a plugin bound to that ABI stops loading the moment TensorFlow rebuilds it.
// The field numbers used here come from graph.proto, node_def.proto and
// attr_value.proto and are fixed by protobuf's compatibility rules.
//
// Everything not understood is carried through byte for byte. A graph this
// pass does not recognise has to come out exactly as it went in, because the
// alternative to a missed fusion is a corrupted model.

/*** READING ***/

class Reader {
 public:
  Reader(const uint8_t* data, size_t size) : p_(data), end_(data + size) {}

  bool done() const { return p_ >= end_; }
  bool ok() const { return ok_; }

  bool Varint(uint64_t* value) {
    uint64_t result = 0;
    int shift = 0;
    while (p_ < end_) {
      const uint8_t byte = *p_++;
      result |= static_cast<uint64_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) {
        *value = result;
        return true;
      }
      shift += 7;
      if (shift > 63) break;
    }
    ok_ = false;
    return false;
  }

  // Reads one field header and its payload, leaving the payload's bounds in
  // `begin` and `length` for length-delimited fields.
  bool Field(int* number, int* wire_type, const uint8_t** begin,
             size_t* length) {
    uint64_t tag = 0;
    if (!Varint(&tag)) return false;
    *number = static_cast<int>(tag >> 3);
    *wire_type = static_cast<int>(tag & 7);
    const uint8_t* start = p_;
    switch (*wire_type) {
      case 0: {  // varint
        uint64_t ignored;
        if (!Varint(&ignored)) return false;
        break;
      }
      case 1:  // 64 bit
        if (end_ - p_ < 8) { ok_ = false; return false; }
        p_ += 8;
        break;
      case 2: {  // length delimited
        uint64_t size = 0;
        if (!Varint(&size)) return false;
        if (static_cast<uint64_t>(end_ - p_) < size) { ok_ = false; return false; }
        *begin = p_;
        *length = static_cast<size_t>(size);
        p_ += size;
        return true;
      }
      case 5:  // 32 bit
        if (end_ - p_ < 4) { ok_ = false; return false; }
        p_ += 4;
        break;
      default:
        ok_ = false;
        return false;
    }
    // Not length delimited: hand back the raw bytes of the whole field so the
    // caller can copy it through untouched.
    *begin = start;
    *length = static_cast<size_t>(p_ - start);
    return true;
  }

  const uint8_t* position() const { return p_; }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
  bool ok_ = true;
};

/*** WRITING ***/

class Writer {
 public:
  void Varint(uint64_t value) {
    while (value >= 0x80) {
      out_.push_back(static_cast<uint8_t>(value) | 0x80);
      value >>= 7;
    }
    out_.push_back(static_cast<uint8_t>(value));
  }

  void Tag(int field, int wire_type) {
    Varint((static_cast<uint64_t>(field) << 3) | wire_type);
  }

  void Bytes(const uint8_t* data, size_t size) {
    out_.insert(out_.end(), data, data + size);
  }

  void StringField(int field, const std::string& value) {
    if (value.empty()) return;
    Tag(field, 2);
    Varint(value.size());
    out_.insert(out_.end(), value.begin(), value.end());
  }

  void LengthDelimited(int field, const uint8_t* data, size_t size) {
    Tag(field, 2);
    Varint(size);
    Bytes(data, size);
  }

  void MessageField(int field, const Writer& message) {
    LengthDelimited(field, message.out_.data(), message.out_.size());
  }

  void Int64Field(int field, int64_t value) {
    Tag(field, 0);
    Varint(static_cast<uint64_t>(value));
  }

  std::vector<uint8_t>& bytes() { return out_; }
  const std::vector<uint8_t>& bytes() const { return out_; }

 private:
  std::vector<uint8_t> out_;
};

/*** THE PIECES OF A NODE THIS PASS LOOKS AT ***/

// A field carried through without being understood.
struct RawField {
  int number = 0;
  int wire_type = 0;
  std::vector<uint8_t> payload;
};

void WriteRaw(const RawField& field, Writer* out) {
  if (field.wire_type == 2) {
    out->LengthDelimited(field.number, field.payload.data(),
                         field.payload.size());
  } else {
    out->Tag(field.number, field.wire_type);
    out->Bytes(field.payload.data(), field.payload.size());
  }
}

struct Node {
  std::string name;
  std::string op;
  std::string device;
  std::vector<std::string> inputs;
  // Attribute entries, keyed by name, each holding the whole map entry's
  // bytes so it can be written back without understanding its value.
  std::map<std::string, std::vector<uint8_t>> attrs;
  // Fields of NodeDef this pass does not model, kept verbatim. The wire type
  // is kept with them: writing a 64-bit field back as a varint would corrupt
  // it, and a pass that rewrites nothing still has to reproduce them exactly.
  std::vector<RawField> other;
  // Set when the node has been folded into another and must not be emitted.
  bool dropped = false;
};

bool ParseNode(const uint8_t* data, size_t size, Node* node) {
  Reader reader(data, size);
  while (!reader.done()) {
    int number = 0, wire_type = 0;
    const uint8_t* begin = nullptr;
    size_t length = 0;
    if (!reader.Field(&number, &wire_type, &begin, &length)) return false;
    switch (number) {
      case 1:
        node->name.assign(reinterpret_cast<const char*>(begin), length);
        break;
      case 2:
        node->op.assign(reinterpret_cast<const char*>(begin), length);
        break;
      case 3:
        node->inputs.emplace_back(reinterpret_cast<const char*>(begin), length);
        break;
      case 4:
        node->device.assign(reinterpret_cast<const char*>(begin), length);
        break;
      case 5: {
        // A map entry: key is field 1, value is field 2. Only the key is read;
        // the entry is stored whole.
        Reader entry(begin, length);
        std::string key;
        while (!entry.done()) {
          int n = 0, w = 0;
          const uint8_t* b = nullptr;
          size_t l = 0;
          if (!entry.Field(&n, &w, &b, &l)) return false;
          if (n == 1 && w == 2) {
            key.assign(reinterpret_cast<const char*>(b), l);
          }
        }
        node->attrs[key].assign(begin, begin + length);
        break;
      }
      default:
        node->other.push_back(
            {number, wire_type, std::vector<uint8_t>(begin, begin + length)});
        break;
    }
  }
  return reader.ok();
}

void WriteNode(const Node& node, Writer* out) {
  Writer body;
  body.StringField(1, node.name);
  body.StringField(2, node.op);
  for (const std::string& input : node.inputs) body.StringField(3, input);
  body.StringField(4, node.device);
  for (const auto& attr : node.attrs) {
    body.LengthDelimited(5, attr.second.data(), attr.second.size());
  }
  for (const RawField& extra : node.other) WriteRaw(extra, &body);
  out->MessageField(1, body);
}

/*** ATTRIBUTES THIS PASS BUILDS ***/

// One map entry holding an integer attribute.
std::vector<uint8_t> IntAttr(const std::string& key, int64_t value) {
  Writer attr_value;
  attr_value.Int64Field(3, value);  // AttrValue.i
  Writer entry;
  entry.StringField(1, key);
  entry.MessageField(2, attr_value);
  return entry.bytes();
}

// One map entry holding a list of one type. _FusedConv2D declares its extra
// inputs as "args: TArgs" with no default, so the fusion has to say what type
// they are; it is the convolution's own.
std::vector<uint8_t> TypeListAttr(const std::string& key, uint64_t type) {
  Writer list;
  Writer packed;
  packed.Varint(type);
  // Packed repeated enums are one length-delimited field of concatenated
  // varints, which for a single entry is that entry's bytes.
  list.LengthDelimited(6, packed.bytes().data(), packed.bytes().size());
  Writer attr_value;
  attr_value.MessageField(1, list);
  Writer entry;
  entry.StringField(1, key);
  entry.MessageField(2, attr_value);
  return entry.bytes();
}

// The DataType out of an attribute entry holding a single type, which is what
// a "T" attribute is. Returns false for anything else.
bool TypeOfAttr(const std::vector<uint8_t>& entry, uint64_t* type) {
  Reader outer(entry.data(), entry.size());
  while (!outer.done()) {
    int number = 0, wire_type = 0;
    const uint8_t* begin = nullptr;
    size_t length = 0;
    if (!outer.Field(&number, &wire_type, &begin, &length)) return false;
    if (number != 2 || wire_type != 2) continue;  // the AttrValue
    Reader value(begin, length);
    while (!value.done()) {
      int n = 0, w = 0;
      const uint8_t* b = nullptr;
      size_t l = 0;
      if (!value.Field(&n, &w, &b, &l)) return false;
      if (n == 6 && w == 0) {  // AttrValue.type
        Reader number_reader(b, l);
        return number_reader.Varint(type);
      }
    }
  }
  return false;
}

// One map entry holding a list of strings.
std::vector<uint8_t> StringListAttr(const std::string& key,
                                    const std::vector<std::string>& values) {
  Writer list;
  for (const std::string& value : values) list.StringField(2, value);
  Writer attr_value;
  attr_value.MessageField(1, list);  // AttrValue.list
  Writer entry;
  entry.StringField(1, key);
  entry.MessageField(2, attr_value);
  return entry.bytes();
}

/*** THE FUSION ***/

// The activations the fused kernels implement. Anything else stops the fusion
// at the bias, which is still worth doing.
bool IsFusableActivation(const std::string& op) {
  return op == "Relu" || op == "Relu6" || op == "Elu" || op == "LeakyRelu" ||
         op == "Sigmoid" || op == "Tanh";
}

// The producer named by an input, with the ":0" and the "^" stripped. Returns
// an empty string for a control dependency, which never counts as a consumer
// of a value and must not be fused across.
std::string ProducerOf(const std::string& input, bool* is_control) {
  *is_control = !input.empty() && input[0] == '^';
  if (*is_control) return input.substr(1);
  const size_t colon = input.find(':');
  return colon == std::string::npos ? input : input.substr(0, colon);
}

// The attributes each fused op declares. Anything else the original node
// carried has to be dropped: MatMul has grad_a and grad_b, which _FusedMatMul
// does not declare, and a node carrying an undeclared attribute is reported
// on every run. Attributes whose name begins with an underscore are
// TensorFlow's own bookkeeping and are always allowed.
bool AttrBelongs(const std::string& name, bool is_conv) {
  if (!name.empty() && name[0] == '_') return true;
  static const std::set<std::string>* conv_attrs = new std::set<std::string>{
      "T",           "TArgs",         "num_args",      "num_host_args",
      "strides",     "padding",       "explicit_paddings", "data_format",
      "filter_format", "dilations",   "use_cudnn_on_gpu", "fused_ops",
      "epsilon",     "leakyrelu_alpha"};
  static const std::set<std::string>* matmul_attrs = new std::set<std::string>{
      "T",        "transpose_a", "transpose_b", "num_args",
      "fused_ops", "epsilon",    "leakyrelu_alpha"};
  const std::set<std::string>& allowed = is_conv ? *conv_attrs : *matmul_attrs;
  return allowed.count(name) != 0;
}

bool OnThisDevice(const std::string& device) {
  // An unassigned node is still a candidate: placement runs after this, and a
  // graph handed to this optimizer is one TensorFlow has already decided
  // belongs to this device type.
  return device.empty() || device.find("GPU") != std::string::npos;
}

}  // namespace

bool FuseGraph(const uint8_t* data, size_t size, std::vector<uint8_t>* out) {
  // Top level pass: nodes are collected, everything else is kept verbatim in
  // the order it appeared.
  std::vector<Node> nodes;
  std::vector<RawField> other;
  Reader reader(data, size);
  while (!reader.done()) {
    int number = 0, wire_type = 0;
    const uint8_t* begin = nullptr;
    size_t length = 0;
    if (!reader.Field(&number, &wire_type, &begin, &length)) return false;
    if (number == 1 && wire_type == 2) {
      Node node;
      if (!ParseNode(begin, length, &node)) return false;
      nodes.push_back(std::move(node));
    } else {
      other.push_back(
          {number, wire_type, std::vector<uint8_t>(begin, begin + length)});
    }
  }
  if (!reader.ok()) return false;

  // How many times each node is named as an input, and by whom. A tensor with
  // more than one consumer cannot be folded away: the other consumers would
  // lose it.
  std::map<std::string, int> uses;
  std::map<std::string, int> index_of;
  for (size_t i = 0; i < nodes.size(); ++i) index_of[nodes[i].name] = i;
  for (const Node& node : nodes) {
    for (const std::string& input : node.inputs) {
      bool control = false;
      uses[ProducerOf(input, &control)]++;
    }
  }

  // Fused nodes are collected rather than appended while the loop runs: a
  // push_back into the vector being iterated would move every node and leave
  // every reference and index above pointing at freed memory.
  std::vector<Node> additions;
  int fused = 0;
  for (size_t bias_index = 0; bias_index < nodes.size(); ++bias_index) {
    Node& bias = nodes[bias_index];
    if (bias.op != "BiasAdd" || bias.dropped) continue;
    if (bias.inputs.size() != 2) continue;
    bool control = false;
    const std::string producer = ProducerOf(bias.inputs[0], &control);
    if (control) continue;
    auto found = index_of.find(producer);
    if (found == index_of.end()) continue;
    Node& base = nodes[found->second];
    if (base.dropped) continue;
    const bool is_conv = base.op == "Conv2D";
    const bool is_matmul = base.op == "MatMul";
    if (!is_conv && !is_matmul) continue;
    // The convolution's output must feed the bias and nothing else, or the
    // other consumers would be left pointing at a node that no longer exists.
    if (uses[base.name] != 1) continue;
    if (!OnThisDevice(base.device) || !OnThisDevice(bias.device)) continue;
    // A MatMul with transposed operands has a different fused kernel contract
    // than the plain one, so only the shapes the fused kernel reads are taken.
    if (base.attrs.count("T") == 0) continue;

    // An activation directly after, if it is one of the fusable ones and the
    // bias likewise feeds nothing else.
    size_t activation_index = nodes.size();  // "none"
    if (uses[bias.name] == 1) {
      for (size_t i = 0; i < nodes.size(); ++i) {
        const Node& candidate = nodes[i];
        if (candidate.dropped || !IsFusableActivation(candidate.op)) continue;
        if (candidate.inputs.size() != 1) continue;
        bool candidate_control = false;
        if (ProducerOf(candidate.inputs[0], &candidate_control) != bias.name ||
            candidate_control) {
          continue;
        }
        if (!OnThisDevice(candidate.device)) break;
        activation_index = i;
        break;
      }
    }
    Node* activation =
        activation_index < nodes.size() ? &nodes[activation_index] : nullptr;

    // Build the fused node in place of the base, keeping its name so every
    // other reference in the graph stays valid.
    Node fused_node = base;
    fused_node.op = is_conv ? "_FusedConv2D" : "_FusedMatMul";
    for (auto it = fused_node.attrs.begin(); it != fused_node.attrs.end();) {
      it = AttrBelongs(it->first, is_conv) ? std::next(it)
                                           : fused_node.attrs.erase(it);
    }
    fused_node.inputs.push_back(bias.inputs[1]);
    // Control dependencies on the folded nodes have to survive the fold.
    for (const Node* folded : {static_cast<const Node*>(&bias),
                               static_cast<const Node*>(activation)}) {
      if (folded == nullptr) continue;
      for (const std::string& input : folded->inputs) {
        bool folded_control = false;
        ProducerOf(input, &folded_control);
        if (folded_control) fused_node.inputs.push_back(input);
      }
    }
    fused_node.attrs["num_args"] = IntAttr("num_args", 1);
    std::vector<std::string> ops = {"BiasAdd"};
    if (activation != nullptr) ops.push_back(activation->op);
    fused_node.attrs["fused_ops"] = StringListAttr("fused_ops", ops);
    if (is_conv) {
      // TArgs has no default, so a fused convolution that does not name it is
      // rejected when the graph is next validated.
      uint64_t type = 0;
      if (!TypeOfAttr(base.attrs["T"], &type)) continue;
      fused_node.attrs["TArgs"] = TypeListAttr("TArgs", type);
    }

    // The node that keeps the name the rest of the graph refers to is the
    // last one folded, so that references to it still resolve.
    const std::string keep = activation != nullptr ? activation->name
                                                   : bias.name;
    fused_node.name = keep;
    base.dropped = true;
    bias.dropped = true;
    if (activation != nullptr) activation->dropped = true;
    additions.push_back(std::move(fused_node));
    ++fused;
  }

  Writer writer;
  for (const Node& node : nodes) {
    if (node.dropped) continue;
    WriteNode(node, &writer);
  }
  for (const Node& node : additions) WriteNode(node, &writer);
  for (const RawField& extra : other) WriteRaw(extra, &writer);
  *out = std::move(writer.bytes());
  if (fused > 0 && std::getenv("TF_METAL_LOG_FUSION") != nullptr) {
    LOG(INFO) << "Metal: fused " << fused << " bias and activation chains.";
  }
  return true;
}

namespace {

void Optimize(void* optimizer, const TF_Buffer* input,
              const TF_GrapplerItem* item, TF_Buffer* output,
              TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  std::vector<uint8_t> rewritten;
  const auto* data = static_cast<const uint8_t*>(input->data);
  if (!FuseGraph(data, input->length, &rewritten)) {
    // A graph this pass could not read is passed through unchanged. Refusing
    // to optimise is a missed fusion; guessing is a broken model.
    LOG(WARNING) << "Metal: could not read the graph to optimise it; passing "
                    "it through unchanged.";
    rewritten.assign(data, data + input->length);
  }
  auto* buffer = new uint8_t[rewritten.size()];
  std::memcpy(buffer, rewritten.data(), rewritten.size());
  output->data = buffer;
  output->length = rewritten.size();
  output->data_deallocator = [](void* memory, size_t) {
    delete[] static_cast<uint8_t*>(memory);
  };
}

}  // namespace

void MetalInitGraph(TP_OptimizerRegistrationParams* params,
                    TF_Status* status) {
  params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->major_version = GO_MAJOR;
  params->minor_version = GO_MINOR;
  params->patch_version = GO_PATCH;
  params->device_type = kMetalDeviceType;

  params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;
  params->optimizer->create_func = nullptr;
  params->optimizer->optimize_func = &Optimize;
  params->optimizer->destroy_func = nullptr;

  params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  // The layout optimizer rewrites NHWC convolutions into NCHW ones with a
  // transpose on either side. That pays for itself on a device whose
  // convolution kernels want NCHW; MPSGraph takes either, so here the
  // transposes are pure loss. Measured on a 4x16x16x8 convolution they cost
  // 42.8 and 44.2 microseconds around a 43.0 microsecond convolution, close
  // to tripling the work.
  params->optimizer_configs->layout_optimizer = TF_TriState_Off;
  TF_SetStatus(status, TF_OK, "");
}

}  // namespace metal
}  // namespace tensorflow
