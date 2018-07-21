/* 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_ROCM
#ifndef TENSORFLOW_RTGLIB_CONVERT_
#define TENSORFLOW_RTGLIB_CONVERT_

#ifndef TENSORFLOW_RTGLIB_COMMON_HEADER_
#include "common_headers.h"
#endif  // TENSORFLOW_RTGLIB_COMMON_HEADER_
#include "rocm/include/migraph/operation.hpp"
#include "rocm/include/migraph/instruction.hpp"
#include "rocm/include/migraph/program.hpp"
#include "rocm/include/migraph/operators.hpp"
#include "rocm/include/migraph/generate.hpp"
#include "rocm/include/migraph/iterator_for.hpp"
#include "rocm/include/migraph/cpu/cpu_target.hpp"
#include "rocm/include/migraph/miopen/target.hpp"
#include "rocm/include/migraph/miopen/miopen.hpp"
#include "rocm/include/migraph/miopen/hip.hpp"
#include "rocm/include/miopen/miopen.h"

#define ALIGN_BYTES(addr, size) (((addr % size) != 0) ? (addr += (size - addr % size)) : addr)

namespace tensorflow {
namespace rtglib {
namespace convert {

#define MIN_CLUSTER_SIZE 2

typedef enum {
    is_entry = 0,
    is_exit
} NodeMask;    

struct Cluster {
     explicit Cluster() { init(); }
     void addInputEdge(const Edge* edge)  { input_edges.push_back(edge);  }
     void addOutputEdge(const Edge* edge) { output_edges.push_back(edge); }
     void addNode(Node* node)             { nodes.push_back(node);        }
     int  getSize()                       { return nodes.size();          }
     std::vector<const Edge*> input_edges;
     std::vector<const Edge*> output_edges;
     std::vector<Node*> nodes;  // sorted in reversed post order.
     void init() {
         input_edges.clear();
         output_edges.clear();
         nodes.clear();
     }
};

typedef migraph::instruction_ref T_RTG_INST_REF;
typedef std::vector<T_RTG_INST_REF> T_RTG_INST_REFS; 
typedef const std::vector<std::pair<string, Tensor>> T_INPUT_MAP; 
 
class  Converter;
using OpConverter =
    std::function<tensorflow::Status(Converter&, const tensorflow::NodeDef&, const T_RTG_INST_REFS&)>;

using AttrEncoder=
    std::function<void(migraph::instruction&, NameAttrList&, Converter&)>;

using AttrDecoder=
    std::function<void(const NameAttrList&, Converter*, string&)>;
    
struct Converter {
    explicit Converter(migraph::program* p, T_INPUT_MAP* map) {
        Init(); program = p; inputs = map;
    }
    bool isCandidate(const Node*);
    bool isRegistered(const Node*);
    void add_instruction(const Node*, bool);
    void add_parameter(const NodeDef&, TensorShapeProto&);
    int get_offset(int bytes, int ele_size) {
        int cur_offset = ALIGN_BYTES(next_offset, ele_size);
        next_offset = cur_offset + bytes;
        return cur_offset;
    }
    int get_offset(migraph::shape shape) {
        int bytes = shape.bytes();
        int ele_size = bytes/shape.elements();
        return get_offset(bytes, ele_size);
    }
    T_RTG_INST_REF add_transpose(const T_RTG_INST_REFS&, int, std::vector<int64_t>&);
    void decodeAttr(const NameAttrList&);
    void getNodeType(const NodeDef&, DataType*);
    bool getNCHWFormat(const T_RTG_INST_REFS&);
    migraph::shape getNodeShape(const NodeDef&, DataType* p_dtype = nullptr, TensorShapeProto* proto = nullptr);
    migraph::shape getAttrShape(const NameAttrList&);
    migraph::shape::type_t getShapeType(const DataType&);
    migraph::shape getShape(const Tensor*);
    DataType getType(const migraph::shape::type_t&);
    void getTensorShape(const migraph::shape&, TensorShape&);
    void getLiteralFromTensor(const TensorProto&, migraph::literal&);
    void getLiteral(migraph::shape, const char*, int, migraph::literal&);
    std::unordered_map<string, OpConverter> op_registry_;
    std::unordered_map<string, AttrEncoder> attr_encoder_registry_;
    std::unordered_map<string, AttrDecoder> attr_decoder_registry_;
    void Init() {
        next_offset = 0;
        register_op_converters();
        register_attr_encoders();
        register_attr_decoders();
        instructions.clear();
        rtgInsNames.clear();
        rtgInsCnt.clear();
        rtgInsOutputFormat.clear();
    }
    string lookupEncoder(const string);
    string lookupDecoder(const string);
    void register_op_converters();
    void register_attr_encoders();
    void register_attr_decoders();
    bool starts_with(const string&, const string&);
    bool contains(const string&, const string&);
    string substract_prefix(const string&, const string&);
    std::unordered_map<string, T_RTG_INST_REF> instructions;
    std::unordered_map<migraph::instruction*, string> rtgInsNames;
    std::unordered_map<migraph::instruction*, string> rtgInsOutputFormat;
    std::unordered_map<string, int> rtgInsCnt;
    migraph::program* program;
    T_INPUT_MAP* inputs;
    int next_offset;
    static const string prefix;
    static const string postfix;
    static const string param_prefix;
    static const string literal_prefix;
    static const string gpuDeviceSubStr;
    string device;
};

const string Converter::prefix = "@";
const string Converter::postfix = "@";
const string Converter::param_prefix = "@param";
const string Converter::literal_prefix = "@literal";
const string Converter::gpuDeviceSubStr = "GPU";

 
Status AddActivation(Converter&, const NodeDef&, const T_RTG_INST_REFS&);
Status AddBiasAdd(Converter&, const NodeDef&, const T_RTG_INST_REFS&);
Status AddConst(Converter&, const NodeDef&, const T_RTG_INST_REFS&);
Status AddConv2D(Converter&, const NodeDef&, const T_RTG_INST_REFS&);
Status AddIdentity(Converter&, const NodeDef&, const T_RTG_INST_REFS&);
Status AddMaxPool(Converter&, const NodeDef&, const T_RTG_INST_REFS&);
Status AddScale(Converter&, const NodeDef&, const T_RTG_INST_REFS&);
Status ConvertGraphToRTG(std::unique_ptr<Graph>*, T_INPUT_MAP*);
Status ConvertSubGraphToRTG(std::unique_ptr<Graph>*, Cluster&, T_INPUT_MAP*, std::unordered_map<int, unsigned>&, bool, ShapeRefiner&);
Status BuildLaunchNode(std::unique_ptr<Graph>*, Cluster&,Converter&, string&);
void SetInputAttr(migraph::instruction&, NameAttrList&, Converter&);
void SetNameAttr(migraph::instruction&, NameAttrList&, Converter&); 
void EncodeActivationAttr(migraph::instruction&, NameAttrList&, Converter&);
void EncodeConstAttr(migraph::instruction&, NameAttrList&, Converter&); 
void EncodeConvolutionAttr(migraph::instruction&, NameAttrList&, Converter&);
void EncodeParamAttr(migraph::instruction&, NameAttrList&, Converter&);
void EncodeTransposeAttr(migraph::instruction&, NameAttrList&, Converter&);
void EncodeContiguousAttr(migraph::instruction&, NameAttrList&, Converter&);
void EncodePoolingAttr(migraph::instruction&, NameAttrList&, Converter&);
void DecodeActivationAttr(const NameAttrList&, Converter*, string&);
void DecodeConstAttr(const NameAttrList&, Converter*, string&);
void DecodeConvolutionAttr(const NameAttrList&, Converter*, string&);
void DecodeTransposeAttr(const NameAttrList&, Converter*, string&);
void DecodeContiguousAttr(const NameAttrList&, Converter*, string&); 
void DecodeInputAttr(T_RTG_INST_REFS& inputs, const NameAttrList& func, Converter* convert);
void DecodeParamAttr(const NameAttrList&, Converter*, string&); 
void DecodePoolingAttr(const NameAttrList&, Converter*, string&);
} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_RTGLIB_CONVERT_
#endif // TENSORFLOW_USE_ROCM
