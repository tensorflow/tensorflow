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

#include "convert_graph.h"
#include "dump_graph.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include <stack>
#include <unordered_map>

#define RTGLIB tensorflow::rtglib

namespace tensorflow {
namespace rtglib {
namespace convert {

Status AddConv2D(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_REFS& inputs) {
    migraph::convolution op;
    string data_format;
    TF_RETURN_IF_ERROR(GetNodeAttr(nodeDef, "data_format", &data_format));
    int h_index = 2;
    int w_index = 3;
    int filter_row_index = 0;
    int filter_col_index = 1;
    bool addInputTranspose = false;
    // MIGraph use "NCHW" as default.
    if (ctx.starts_with(data_format, "NHWC")) {
        h_index = 1;
        w_index = 2;
        T_RTG_INST_REF input0 = inputs[0];
        migraph::instruction* ptr = &(*input0);
        if (ctx.rtgInsOutputFormat.find(ptr) == ctx.rtgInsOutputFormat.end()) {
            addInputTranspose = true;
        } else {
            CHECK(ctx.rtgInsOutputFormat[ptr] == "NCHW") << "unexpected input format";
        }
    }
    
    auto list = nodeDef.attr().at("strides").list();
    std::vector<int> strides;
    int stride_rows = list.i(h_index);
    int stride_cols = list.i(w_index);
    strides.push_back(stride_rows);
    strides.push_back(stride_cols);
    std::copy(strides.begin(), strides.end(), op.stride.begin());

    int count = 0;
    int input_rows, input_cols;
    int filter_rows, filter_cols;
    Padding padding;
    TF_RETURN_IF_ERROR(GetNodeAttr(nodeDef, "padding", &padding));
    switch (padding) {
    case Padding::VALID:
        op.padding_mode = migraph::convolution::valid;
        break;
    case Padding::SAME:
        op.padding_mode = migraph::convolution::same;
        break;
    };

    T_RTG_INST_REFS new_inputs;
    for (auto iter = inputs.begin(), end = inputs.end(); iter != end; iter++) {
        T_RTG_INST_REF ins = *iter;
        migraph::shape shape = ins->result;
        TensorShape tensor_shape;
        ctx.getTensorShape(shape, tensor_shape);
        if (count == 0) {
            // input
            // batch_size = tensor_shape.dim_size(0);
            input_rows = tensor_shape.dim_size(h_index);
            input_cols = tensor_shape.dim_size(w_index);
        } else {
            // filter
            filter_rows = tensor_shape.dim_size(filter_row_index);
            filter_cols = tensor_shape.dim_size(filter_col_index);
        }
        new_inputs.push_back(ins);
        count++;
    }
    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    TF_RETURN_IF_ERROR(GetWindowedOutputSize(input_rows, filter_rows,
                                             stride_rows,
                                             padding, &out_rows, &pad_rows));
    TF_RETURN_IF_ERROR(GetWindowedOutputSize(input_cols, filter_cols,
                                             stride_cols,
                                             padding, &out_cols, &pad_cols));
    std::vector<int> paddings;
    paddings.push_back(pad_rows);
    paddings.push_back(pad_cols);
    std::copy(paddings.begin(), paddings.end(), op.padding.begin());

    if (nodeDef.attr().find("dilations") != nodeDef.attr().end()) {
        auto list = nodeDef.attr().at("dilations").list();
        std::vector<int> dilations;
        for (int i = 0; i < list.i_size(); ++i)
            dilations.push_back(list.i(i));
        std::copy(dilations.begin(), dilations.end(), op.dilation.begin());
    }
    
    if (addInputTranspose) {
        // Transpose input.
        std::vector<int64_t> perm0 = {0, 3, 1, 2};
        new_inputs[0] = ctx.add_transpose(inputs, 0, perm0);
    }
    
    // Transpose filter.
    std::vector<int64_t> perm1 = {3, 2, 0, 1};
    new_inputs[1] = ctx.add_transpose(inputs, 1, perm1);
    T_RTG_INST_REF new_ins = ctx.program->add_instruction(op, new_inputs);
    ctx.instructions[nodeDef.name()] = new_ins;

    if (addInputTranspose)
        ctx.rtgInsOutputFormat[&(*new_ins)] = "NCHW";
    
    return Status::OK();
}

Status AddMaxPool(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_REFS& inputs) {
    bool nchw = ctx.getNCHWFormat(inputs);
    T_RTG_INST_REF ins = ctx.program->add_instruction(migraph::pooling{"max"}, inputs);
    ctx.instructions[nodeDef.name()] = ins;
    if (nchw)
        ctx.rtgInsOutputFormat[&(*ins)] = "NCHW";
    return Status::OK();
}

Status AddBiasAdd(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_REFS& inputs) {
    CHECK(false);
    return Status::OK();
}

Status AddConst(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_REFS& inputs) {
    const auto& tensor = nodeDef.attr().at("value").tensor();
    auto& content = tensor.tensor_content();
    DataType dataType;
    ctx.getNodeType(nodeDef, &dataType);
    migraph::shape shape = ctx.getNodeShape(nodeDef, &dataType);
    migraph::literal li;
    const char* data_ptr = reinterpret_cast<const char*>(content.data());
    ctx.getLiteral(shape, data_ptr, content.size(), li);
    ctx.instructions[nodeDef.name()] = ctx.program->add_literal(li);
    return Status::OK();
}

Status AddIdentity(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_REFS& inputs) {
    CHECK(false);
    return Status::OK();
}

Status AddActivation(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_REFS& inputs) {
    bool nchw = ctx.getNCHWFormat(inputs);
    T_RTG_INST_REF ins = ctx.program->add_instruction(migraph::activation{"relu"}, inputs);
    ctx.instructions[nodeDef.name()] = ins;
    if (nchw)
        ctx.rtgInsOutputFormat[&(*ins)] = "NCHW";
    return Status::OK();
}

Status AddScale(Converter& ctx, const NodeDef& nodeDef, const T_RTG_INST_REFS& inputs) {
    CHECK(false);
    return Status::OK();
}

void Converter::register_op_converters()  {
    op_registry_["Const"] = AddConst;
    op_registry_["Conv2D"] = AddConv2D;
    op_registry_["Relu"] = AddActivation;
#if 0
    op_registry_["BiasAdd"] = AddScale;
    op_registry_["MaxPool"] = AddMaxPool;
    op_registry_["Identity"] = AddIdentity;
#endif    
}

void Converter::register_attr_encoders() {
    attr_encoder_registry_["@param"] = EncodeParamAttr;
    attr_encoder_registry_["@literal"] = EncodeConstAttr;
    attr_encoder_registry_["convolution"] = EncodeConvolutionAttr;
    attr_encoder_registry_["activation"] = EncodeActivationAttr;
    attr_encoder_registry_["transpose"] = EncodeTransposeAttr;
    attr_encoder_registry_["contiguous"] = EncodeContiguousAttr;
    attr_encoder_registry_["pooling"] = EncodePoolingAttr;
}

void Converter::register_attr_decoders() {
    attr_decoder_registry_["@param"] = DecodeParamAttr;
    attr_decoder_registry_["@literal"] = DecodeConstAttr;
    attr_decoder_registry_["convolution"] = DecodeConvolutionAttr;
    attr_decoder_registry_["activation"] = DecodeActivationAttr;
    attr_decoder_registry_["transpose"] = DecodeTransposeAttr;
    attr_decoder_registry_["contiguous"] = DecodeContiguousAttr;
    attr_decoder_registry_["pooling"] = DecodePoolingAttr;
}
    
bool Converter::starts_with(const string& value, const string& prefix)
{
    if (prefix.size() <= value.size()) {
        return std::equal(prefix.begin(), prefix.end(), value.begin());
    }
    return false;
}

bool Converter::contains(const string& str1, const string& str2)
{
    return (std::search(str1.begin(), str1.end(), str2.begin(), str2.end()) != str1.end());
}

string Converter::substract_prefix(const string& value, const string& prefix) {
    CHECK(prefix.length() < value.length()) << "unexpected prefix";
    string str = value.substr(prefix.length());
    if (starts_with(str, ":"))
        str = str.substr(1);
#if 0
    if (starts_with(str, Converter::prefix)) {
        str = str.substr(Converter::prefix.length());
        std::size_t pos = str.find(Converter::postfix);
        CHECK(pos != std::string::npos) << "Postfix unfound";
        str = str.substr(pos + Converter::postfix.length());
    }
#endif    
    return str;
}

string Converter::lookupEncoder(const string name)
{
    for (auto iter = attr_encoder_registry_.begin(); iter != attr_encoder_registry_.end(); ++iter) {
        string rtg_name = iter->first;
        if (starts_with(name, rtg_name))
            return rtg_name;
    }
    return "";
}

string Converter::lookupDecoder(const string name)
{
    for (auto iter = attr_decoder_registry_.begin(); iter != attr_decoder_registry_.end(); ++iter) {
        string rtg_name = iter->first;
        if (starts_with(name, rtg_name))
            return rtg_name;
    }
    return "";
}    
    
bool Converter::isRegistered(const Node * node) {
    return op_registry_.count(node->type_string());
}

bool Converter::isCandidate(const Node * node) {
    if (!node->IsOp() || !isRegistered(node))
        return false;
    const NodeDef& nodeDef = node->def();
    // If user has specified a non-GPU device.
    if (!nodeDef.device().empty() && !contains(nodeDef.device(), Converter::gpuDeviceSubStr))
        return false;
    // If runtime has assigned a non-GPU device.
    if (!node->assigned_device_name().empty() && !contains(node->assigned_device_name(), Converter::gpuDeviceSubStr))
        return false;
    return true;
}

void Converter::add_parameter(const NodeDef& nodeDef, TensorShapeProto &proto)  {
    const migraph::shape shape = getNodeShape(nodeDef, nullptr, &proto);
    const string& name = nodeDef.name();
    instructions[name] = program->add_parameter(name, shape);
}

void Converter::add_instruction(const Node* node, bool isExit)  {
    OpConverter op_converter = op_registry_.at(node->type_string());
    T_RTG_INST_REFS inputs;
    std::vector<const Edge*> input_edges;
    input_edges.resize(node->num_inputs(), nullptr);
    for (const Edge* edge : node->in_edges()) {
        if (edge->IsControlEdge()) {
            input_edges.push_back(edge);
        } else {
            input_edges[edge->dst_input()] = edge;
        }
    }
    for (auto iter = input_edges.begin(), end = input_edges.end(); iter != end; iter++) {
        const Edge* edge = *iter;
        if (edge->IsControlEdge())
            continue;
        const string& name = edge->src()->name();
        CHECK(instructions.find(name) != instructions.end()) << "missing input instruction";
        T_RTG_INST_REF input = instructions[name];
        inputs.push_back(input);
    }
    Status s = op_converter(*this, node->def(), inputs);;
    T_RTG_INST_REF ins = std::prev(program->end());
    migraph::instruction* ptr = &(*ins);

    if (isExit && (rtgInsOutputFormat.find(ptr) != rtgInsOutputFormat.end())) {
        CHECK(rtgInsOutputFormat[ptr] == "NCHW") << "Unexpected data format";
        T_RTG_INST_REF ins = std::prev(program->end());
        T_RTG_INST_REFS args;
        args.push_back(ins);
        std::vector<int64_t> perm = {0, 2, 3, 1};
        auto result1 = program->add_instruction(migraph::transpose{perm}, args);
        program->add_instruction(migraph::contiguous{}, result1);
    }

    CHECK(s == Status::OK()) << "fail to add instruction";
}

T_RTG_INST_REF Converter::add_transpose(const T_RTG_INST_REFS& inputs, int index, std::vector<int64_t>& perm) {
    const T_RTG_INST_REF ins = inputs[index];
    if (starts_with(ins->op.name(), Converter::literal_prefix)) {
        migraph::program * eval_program = new migraph::program;
        auto li = eval_program->add_literal(ins->lit);
        auto trans = eval_program->add_instruction(migraph::transpose{perm}, li);
        eval_program->add_instruction(migraph::contiguous{}, trans);
        eval_program->compile(migraph::cpu::cpu_target{});
        std::unordered_map<string, migraph::argument> params;
        migraph::argument arg = eval_program->eval(params);
        migraph::shape arg_shape = arg.get_shape();
        int arg_size = arg_shape.bytes();
        migraph::literal new_li;
        const char * data_ptr = arg.cast<char>();
        getLiteral(arg_shape, data_ptr, arg_size, new_li);
        T_RTG_INST_REF new_ins = program->add_literal(new_li);
        delete eval_program;
        return new_ins;
    } else {
        auto result = program->add_instruction(migraph::transpose{perm}, ins);
        return program->add_instruction(migraph::contiguous{}, result);
    }
}
    
void Converter::decodeAttr(const NameAttrList& func)
{
    string name = func.name();
    string rtg_name = lookupDecoder(name);
    if (rtg_name != "") {
        AttrDecoder attr_decoder = attr_decoder_registry_.at(rtg_name);
        attr_decoder(func, this, rtg_name);
    } else {
        CHECK(false) << "Unknown RTG instruction";
    }
}

DataType Converter::getType(const migraph::shape::type_t& shape_type)
{
    switch (shape_type) {
    case migraph::shape::float_type: return DT_FLOAT; break;
    case migraph::shape::double_type: return DT_DOUBLE; break;
    case migraph::shape::int64_type: return DT_INT64; break;
    case migraph::shape::int32_type: return DT_INT32; break;
    case migraph::shape::int16_type: return DT_INT16; break;
    case migraph::shape::uint16_type: return DT_UINT16; break;
    case migraph::shape::int8_type: return DT_INT8; break;
    default:
        CHECK(false) << "unmatched RTG data type";
    }
}

void Converter::getNodeType(const NodeDef& nodeDef, DataType* data_type)
{
    if (nodeDef.attr().count("dtype")) {
        GetNodeAttr(nodeDef, "dtype", data_type);
    } else if (nodeDef.attr().count("T")) {
        GetNodeAttr(nodeDef, "T", data_type);
    } else {
        CHECK(false) << "data type not found";
    }
}

bool Converter::getNCHWFormat(const T_RTG_INST_REFS& inputs) {
    bool nchw = false;
    for (auto iter = inputs.begin(), end = inputs.end(); iter != end; iter++) {
        T_RTG_INST_REF ins = *iter;
        migraph::instruction* ptr = &(*ins);
        if (rtgInsOutputFormat.find(ptr) != rtgInsOutputFormat.end()) {
            CHECK(rtgInsOutputFormat[ptr] == "NCHW") << "unexpected input format";
            nchw = true;
        }
    }
    return nchw;
}
    
migraph::shape::type_t Converter::getShapeType(const DataType& data_type)
{
    switch (data_type) {
    case DT_FLOAT: return migraph::shape::float_type; break;
    case DT_DOUBLE: return migraph::shape::double_type; break;
    case DT_INT64: return migraph::shape::int64_type; break;
    // case DT_UINT64: return migraph::shape::uint64_type; break;
    case DT_INT32: return migraph::shape::int32_type; break;
    //    case DT_UINT32: return migraph::shape::uint32_type; break;
    case DT_INT16: return migraph::shape::int16_type; break;
    case DT_UINT16: return migraph::shape::uint16_type; break;
    case DT_INT8: return migraph::shape::int8_type; break;
    default:
        CHECK(false) << "unmatched RTG data type";
    }
}

migraph::shape Converter::getAttrShape(const NameAttrList& func)
{
    auto map = func.attr();
    DataType data_type = map.at("dtype").type();
    const TensorShapeProto & shape_proto = map.at("shape").shape();
    std::vector<std::size_t> dims;
    for (const auto& dim_proto : shape_proto.dim()) {
        int size = dim_proto.size();
        dims.push_back(size);
    }
    migraph::shape::type_t shape_type = getShapeType(data_type);
    return {shape_type, dims};
}

migraph::shape Converter::getNodeShape(const NodeDef& nodeDef, DataType *p_dtype, TensorShapeProto* proto) {
    std::string name = nodeDef.name();
    DataType data_type;
    if (p_dtype == nullptr)
        getNodeType(nodeDef, &data_type);
    else
        data_type = *p_dtype;
    migraph::shape::type_t shape_type = getShapeType(data_type);
    std::vector<std::size_t> dims;
    if (nodeDef.attr().count("value")) {
        const TensorProto& raw_val = nodeDef.attr().at("value").tensor();
        DataType d_type = raw_val.dtype();
        CHECK(data_type == d_type) << "data type unmatched";
        const TensorShape& tensor_shape = raw_val.tensor_shape();
        for (int64 i = 0, e = tensor_shape.dims(); i < e; i++)
            dims.push_back(tensor_shape.dim_size(i));
    } else if ((inputs != nullptr) && (nodeDef.op() == "_Arg")) {
        CHECK(nodeDef.attr().count("index")) << "unknown argument index";
        int index;
        GetNodeAttr(nodeDef, "index", &index);
        const Tensor tensor = (*inputs)[index].second;
        const TensorShape& tensor_shape = tensor.shape();
        for (int64 i = 0, e = tensor_shape.dims(); i < e; i++)
            dims.push_back(tensor_shape.dim_size(i));
    } else if (proto != nullptr) {
        for (const auto& dim_proto : proto->dim()) {
            int size = dim_proto.size();
            dims.push_back(size);
        }
    } else {
        CHECK(false) << "unknown shape";
    }
    
    return {shape_type, dims};
}

void Converter::getTensorShape(const migraph::shape& shape, TensorShape& tensor_shape)
{
    const std::vector<std::size_t>& lens = shape.lens();
    int size = lens.size();
    for (int i = 0; i < size; i++)
        tensor_shape.AddDim(lens[i]);
}

migraph::shape Converter::getShape(const Tensor* tensor)
{
    int num_dims = tensor->dims();
    std::vector<std::size_t> dims;
    for (int i = 0; i < num_dims; ++i) {
        int64 dim_size = tensor->dim_size(i);
        dims.push_back(dim_size);
    }
    migraph::shape::type_t shape_type = getShapeType(tensor->dtype());
    migraph::shape shape = {shape_type, dims};
    return shape;
}

void Converter::getLiteral(migraph::shape shape, const char * data_ptr, int size, migraph::literal& li)
{
    DataType data_type = getType(shape.type());
    switch (data_type) {
    case DT_FLOAT: {
        std::vector<float> data;
        int vec_size = size/sizeof(float);
        const float * ptr = reinterpret_cast<const float*>(data_ptr);
        for (int i = 0; i < vec_size; i++)
            data.push_back(ptr[i]);
        li = migraph::literal{shape, data.begin(), data.end()};
        break;
    }
    default:
        CHECK(false) << "unknown data type";
    }
}
    
void Converter::getLiteralFromTensor(const TensorProto& tensor, migraph::literal& li)
{
    const TensorShapeProto& tensor_shape = tensor.tensor_shape();
    auto& content = tensor.tensor_content();
    DataType data_type = tensor.dtype();
    std::vector<std::size_t> dims;
    for (const auto& dim_proto : tensor_shape.dim()) {
        int size = dim_proto.size();
        dims.push_back(size);
    }
    migraph::shape::type_t shape_type = getShapeType(data_type);
    migraph::shape shape = {shape_type, dims};
    const char* data_ptr = reinterpret_cast<const char*>(content.data());
    getLiteral(shape, data_ptr, content.size(), li);
}

void SetNameAttr(migraph::instruction& ins, NameAttrList& attrs, Converter& convert)
{
    string name = ins.op.name();
    int cnt = 0;
    if (convert.rtgInsCnt.find(name) == convert.rtgInsCnt.end()) {
        convert.rtgInsCnt[name] = 0;
    } else {
        cnt = ++(convert.rtgInsCnt[name]);
    }
    string new_name = ins.op.name() + Converter::prefix + std::to_string(cnt) + Converter::postfix;
    attrs.set_name(new_name);
    convert.rtgInsNames[&ins] = new_name;
}
    
void SetInputAttr(migraph::instruction& ins, NameAttrList& attrs, Converter& convert)
{
    auto attr_map = attrs.mutable_attr();
    AttrValue value;
    int32 arg_cnt = ins.arguments.size();
    SetAttrValue(arg_cnt, &value);
    (*attr_map)["num_inputs"] = value;
    arg_cnt = 0;
    for (auto iter = ins.arguments.begin(), end = ins.arguments.end(); iter != end; iter++) {
        T_RTG_INST_REF arg = *iter;
        string name = convert.rtgInsNames[&(*arg)];
        AttrValue value;
        SetAttrValue(name, &value);
        string input_name = "input" + std::to_string(arg_cnt);
        arg_cnt++;
        (*attr_map)[input_name] = value;
    }    
}    

void EncodeActivationAttr(migraph::instruction& ins, NameAttrList& attrs, Converter& convert) {
    SetNameAttr(ins, attrs, convert);
    SetInputAttr(ins, attrs, convert);
}

void EncodePoolingAttr(migraph::instruction& ins, NameAttrList& attrs, Converter& convert) {
    SetNameAttr(ins, attrs, convert);
    SetInputAttr(ins, attrs, convert);
}    
    
void EncodeParamAttr(migraph::instruction& ins, NameAttrList& attrs, Converter& convert) {
    migraph::shape shape = ins.result;
    SetNameAttr(ins, attrs, convert);
    DataType type = convert.getType(shape.type());
    auto attr_map = attrs.mutable_attr();
    AttrValue t_value;
    SetAttrValue(type, &t_value);
    (*attr_map)["dtype"] = t_value;
    TensorShape tensor_shape;
    convert.getTensorShape(shape, tensor_shape);
    AttrValue s_value;
    SetAttrValue(tensor_shape, &s_value);
    (*attr_map)["shape"] = s_value;
}

void EncodeConstAttr(migraph::instruction& ins, NameAttrList& attrs, Converter& convert) {
    SetNameAttr(ins, attrs, convert);
    migraph::shape shape = ins.result;
    DataType type = convert.getType(shape.type());
    auto attr_map = attrs.mutable_attr();
    TensorShape tensor_shape;
    convert.getTensorShape(shape, tensor_shape);
    Tensor tensor(type, tensor_shape);
    int size = tensor.tensor_data().size();
    memcpy(const_cast<char*>(tensor.tensor_data().data()), ins.lit.data(), size);
    TensorProto tensor_proto;
    tensor.AsProtoTensorContent(&tensor_proto);
    AttrValue value;
    SetAttrValue(tensor_proto, &value);
    (*attr_map)["value"] = value;
}

void EncodeConvolutionAttr(migraph::instruction& ins, NameAttrList& attrs, Converter& convert) {
    SetNameAttr(ins, attrs, convert);
    SetInputAttr(ins, attrs, convert);
    migraph::convolution op = migraph::any_cast<migraph::convolution>(ins.op);
    auto* attr_map = attrs.mutable_attr();
    std::vector<int> strides;
    for (auto iter = op.stride.begin(); iter != op.stride.end(); ++iter)
        strides.push_back(*iter);
    SetAttrValue(strides, &(*attr_map)["strides"]);
    std::vector<int> paddings;
    for (auto iter = op.padding.begin(); iter != op.padding.end(); ++iter)
        paddings.push_back(*iter);
    SetAttrValue(paddings, &(*attr_map)["paddings"]);
    std::vector<int> dilations;
    for (auto iter = op.dilation.begin(); iter != op.dilation.end(); ++iter)
        dilations.push_back(*iter);
    SetAttrValue(dilations, &(*attr_map)["dilations"]);
    switch(op.padding_mode) {
    case migraph::convolution::same:
        SetAttrValue("SAME", &(*attr_map)["padding"]);
        break;
    case migraph::convolution::valid:
        SetAttrValue("VALID", &(*attr_map)["padding"]);
    default:
        ;
    }
}

void EncodeTransposeAttr(migraph::instruction& ins, NameAttrList& attr, Converter& convert) {
    SetNameAttr(ins, attr, convert);
    SetInputAttr(ins, attr, convert);
    migraph::transpose op = migraph::any_cast<migraph::transpose>(ins.op);
    auto* attr_map = attr.mutable_attr();
    std::vector<int> dims;
    for (auto iter = op.dims.begin(); iter != op.dims.end(); ++iter)
        dims.push_back(*iter);
    SetAttrValue(dims, &(*attr_map)["permutation"]);
}

void EncodeContiguousAttr(migraph::instruction& ins, NameAttrList& attr, Converter& convert) {
    SetNameAttr(ins, attr, convert);
    SetInputAttr(ins, attr, convert);
}
    
void DecodeActivationAttr(const NameAttrList& func, Converter* convert, string&prefix) {
    string name = func.name();
    T_RTG_INST_REFS inputs;
    DecodeInputAttr(inputs, func, convert);
    convert->instructions[name] = convert->program->add_instruction(migraph::activation{"relu"}, inputs);    
}

void DecodePoolingAttr(const NameAttrList& func, Converter* convert, string&prefix) {
    string name = func.name();
    T_RTG_INST_REFS inputs;
    DecodeInputAttr(inputs, func, convert);
    convert->instructions[name] = convert->program->add_instruction(migraph::pooling{"max"}, inputs);    
}    

void DecodeConstAttr(const NameAttrList& func, Converter* convert, string& prefix) {
    string name = func.name();
    auto map = func.attr();
    const auto& tensor = map.at("value").tensor();
    migraph::literal li;
    convert->getLiteralFromTensor(tensor, li);
    convert->instructions[name] = convert->program->add_literal(li);
    migraph::shape shape = li.get_shape();
    convert->get_offset(shape);
}

void DecodeConvolutionAttr(const NameAttrList& func, Converter* convert, string& prefix) {
    string name = func.name();
    T_RTG_INST_REFS inputs;
    DecodeInputAttr(inputs, func, convert);
    auto map = func.attr();
    migraph::convolution op;
    const auto& list_s = map.at("strides").list();
    std::vector<int> strides;
    for (int i = 0; i < list_s.i_size(); ++i)
        strides.push_back(list_s.i(i));
    std::copy(strides.begin(), strides.end(), op.stride.begin());
    
    const auto& list_p = map.at("paddings").list();
    std::vector<int> paddings;
    for (int i = 0; i < list_p.i_size(); ++i)
        paddings.push_back(list_p.i(i));
    std::copy(paddings.begin(), paddings.end(), op.padding.begin());

    const auto& list_d = map.at("dilations").list();
    std::vector<int> dilations;
    for (int i = 0; i < list_d.i_size(); ++i)
        dilations.push_back(list_d.i(i));
    std::copy(dilations.begin(), dilations.end(), op.dilation.begin());
    const string& padding = map.at("padding").s();
    if (padding == "SAME") {
        op.padding_mode = migraph::convolution::same;
    }
    else if (padding == "VALID") {
        op.padding_mode = migraph::convolution::valid;
    }
    convert->instructions[name] = convert->program->add_instruction(op, inputs);
}

void DecodeTransposeAttr(const NameAttrList& func, Converter* convert, string& prefix) {
    string name = func.name();
    T_RTG_INST_REFS inputs;
    DecodeInputAttr(inputs, func, convert);
    auto map = func.attr();
    std::vector<int64_t> perm;
    const auto& list = map.at("permutation").list();
    for (int i = 0; i < list.i_size(); ++i)
        perm.push_back(list.i(i));
    convert->instructions[name] = convert->program->add_instruction(migraph::transpose{perm}, inputs);
}

void DecodeContiguousAttr(const NameAttrList& func, Converter* convert, string& prefix) {
    string name = func.name();
    T_RTG_INST_REFS inputs;
    DecodeInputAttr(inputs, func, convert);
    convert->instructions[name] = convert->program->add_instruction(migraph::contiguous{}, inputs);
}

void DecodeParamAttr(const NameAttrList& func, Converter* convert, string& prefix) {
    string name = func.name();
    const migraph::shape shape = convert->getAttrShape(func);
    convert->instructions[name] = convert->program->add_parameter(name, shape);
}

void DecodeInputAttr(T_RTG_INST_REFS& inputs, const NameAttrList& func, Converter* convert)
{
    auto map = func.attr();
    int32 num_of_inputs = map.at("num_inputs").i();
    for (int i = 0; i < num_of_inputs; ++i) {
        string input_name = "input" + std::to_string(i);
        string arg_name = map.at(input_name).s();
        CHECK(convert->instructions.find(arg_name) != convert->instructions.end()) << "Input no found";
        inputs.push_back(convert->instructions[arg_name]);
    }
}

Status BuildLaunchNode(std::unique_ptr<Graph>* g, Cluster& cluster, Converter& convert, string& name)
{
    migraph::program* program = convert.program;
    NodeDefBuilder op_builder(name, "RTGLaunchOp");
    std::vector<NodeDefBuilder::NodeOut> income_edges;
    for (const Edge* edge : cluster.input_edges) {
        if (edge->IsControlEdge())
            continue;
        Node* src = edge->src();
        Node* dst = edge->dst();
        int dest_port = edge->dst_input();
        DataType data_type = dst->input_type(dest_port);
        auto income_edge = NodeDefBuilder::NodeOut(src->name(), 
                                                  edge->src_output(), data_type);
        income_edges.emplace_back(income_edge);
    }

    std::vector<DataType> out_types;
    for (const Edge* edge : cluster.output_edges) {
        if (edge->IsControlEdge())
            continue;
        Node* src = edge->src();
        CHECK(!src->IsConstant()) << "TODO: constant is exit node";
        Node* dst = edge->dst();
        int dest_port = edge->dst_input();
        DataType data_type = dst->input_type(dest_port);
        out_types.push_back(data_type);
    }

    gtl::ArraySlice<tensorflow::NodeDefBuilder::NodeOut> input_list(income_edges);
    op_builder.Input(input_list);

    unsigned num_values = 0;
    AttrValue value;
    value.mutable_list()->Clear();
    for (auto ins : migraph::iterator_for(*program)) {
        num_values++;
        NameAttrList& attrs = *(value.mutable_list()->add_func());
        attrs.Clear();
        string name = ins->op.name();
        string rtg_name = convert.lookupEncoder(name);
        if (rtg_name != "") {
            AttrEncoder attr_encoder = convert.attr_encoder_registry_.at(rtg_name);
            attr_encoder(*ins, attrs, convert);
        } else {
            CHECK(false) << "Unknown RTG instruction";
        }
    }

    NameAttrList func;
    func.Clear();
    func.set_name("function");
    (*func.mutable_attr())["func"] = value;
    NodeDef node_def;
    Status status =  op_builder.Attr("function", func)
                        .Attr("OutT", out_types)
                        .Finalize(&node_def);
    CHECK(status.ok()) << "fail to add RTGLaunchOp";
    Graph& graph = **g;
    auto rtg_node = graph.AddNode(node_def, &status);
    TF_RETURN_IF_ERROR(status);

    // Edge info.
    typedef struct {
        Node * src;
        int src_output;
        bool is_control;
    } edge_desc;
    std::vector<edge_desc> input_edges;

    // Cache input edge info.
    for (const Edge* edge : cluster.input_edges)
        input_edges.push_back(edge_desc{edge->src(), edge->src_output(), edge->IsControlEdge()});
    // Construct output edges.
    int ndx = 0;
    for (const Edge* edge : cluster.output_edges) {
        if (!edge->IsControlEdge()) {
            Node* dst = edge->dst();            
            int dest_port = edge->dst_input();    
            TF_RETURN_IF_ERROR(graph.UpdateEdge(rtg_node, ndx, dst, dest_port));
            ndx++;
        }
    }
    // Add control edges at the end.
    for (const Edge* edge : cluster.output_edges) {
        if (edge->IsControlEdge()) {
            Node* dst = edge->dst();
            graph.RemoveEdge(edge);
            graph.AddControlEdge(rtg_node, dst);
        }
    }
    
    // Remove nodes in the subgraph and their edges.
    for (Node* node : cluster.nodes)
        graph.RemoveNode(node);
    // Construct input edges.
    ndx = 0;
    CHECK(rtg_node->in_edges().empty()) << "Unexpected input edges";
    for (auto iter = input_edges.begin(), end = input_edges.end(); iter != end; iter++) {
        if (!(*iter).is_control) {
            graph.AddEdge((*iter).src, (*iter).src_output, rtg_node, ndx);
            ndx++;
        }
    }
    // Add control edges at the end.
    for (auto iter = input_edges.begin(), end = input_edges.end(); iter != end; iter++) {
        if ((*iter).is_control)
            graph.AddControlEdge((*iter).src, rtg_node);
    }
    rtg_node->set_assigned_device_name(convert.device);

#if 0        
    const TensorShapeProto& shape_proto = def->attr().at("shape").shape();
    for (const auto& dim_proto : shape_proto.dim()) {
        size = dim_proto.size();
    }
    protobuf::TextFormat::PrintToString(graph_def, &serialized);
#endif    
    return Status::OK();    
}

Status ConvertSubgraphToRTG(std::unique_ptr<Graph>* g, Cluster& cluster, T_INPUT_MAP * inputs, std::unordered_map<int, unsigned>& id2Mask, bool use_gpu, ShapeRefiner& refiner) {
    migraph::program * program = new migraph::program;
    if (!program)
        return errors::Internal("Fail to create RTG program");

    Converter fwd_convert(program, inputs);
    string param_device;
    for (const Edge* edge : cluster.input_edges) {
        if (edge->IsControlEdge())
            continue;
        Node* src = edge->src();
        param_device = src->assigned_device_name();
        shape_inference::InferenceContext* ic = refiner.GetContext(src);
        TensorShapeProto proto;
        ic->ShapeHandleToProto(ic->output(edge->dst_input()), &proto);
        fwd_convert.add_parameter(src->def(), proto);
    }

    string cluster_name;
    string device;
    for (Node* node : cluster.nodes) {
        bool isExit = (id2Mask[node->id()] & (1 << is_exit)) ? true : false;
        fwd_convert.add_instruction(node, isExit);
        cluster_name += node->name();
        device = node->assigned_device_name();
    }
    std::cout << *program << std::endl;
    // call program->optimize()
    Converter bwd_convert(program, nullptr);
    // TODO: use gpu
    bwd_convert.device = device;
    if (!use_gpu)
        bwd_convert.device = "/job:localhost/replica:0/task:0/cpu:0";
    TF_RETURN_IF_ERROR(BuildLaunchNode(g, cluster, bwd_convert, cluster_name));

    delete program;
    return Status::OK();    
}

Status ConvertGraphToRTG(std::unique_ptr<Graph>* g, T_INPUT_MAP* inputs) {
    const char* env_val = getenv("TF_MIGRAPH_USE_GPU");
    bool use_gpu = false;
    if (env_val != nullptr)
        use_gpu = atoi(env_val);
    const char* cluster_dbg_env = getenv("TF_MIGRAPH_CLUSTER_DBG_LIMIT");
    int cluster_dbg_limit = -1;
    if (cluster_dbg_env != nullptr)
        cluster_dbg_limit = atoi(cluster_dbg_env);
    CHECK_NOTNULL(g);
    const Graph& graph = **g;
    RTGLIB::dump_graph::DumpGraphToFile("Before convert graph to RTG", graph);
    
    std::unordered_map<int, unsigned> id2Order, id2Segment, id2Mask;
    std::unordered_map<int, bool> id2Candidate, id2Visit;
    std::unordered_map<unsigned, unsigned> segment2Cluster;
    unsigned maxNodeNum = 0;
    unsigned maxSegmentNum = 0;
    unsigned maxClusterNum = 0;
    std::vector<Node *> rpOrder;
    GetReversePostOrder(graph, &rpOrder);
    Converter convert(nullptr, nullptr);
    ShapeRefiner refiner(graph.versions().producer(), graph.op_registry());
    
    for (Node* n : rpOrder) {
        int id = n->id();
        id2Order[id] = maxNodeNum++;
        id2Candidate[id] = convert.isCandidate(n) ? true : false;
        id2Mask[id] = 0;
        refiner.AddNode(n);
    }

    Node * sinkNode = graph.sink_node();
    CHECK_NOTNULL(sinkNode);
    std::stack<const tensorflow::Node*> iterStack;
    iterStack.push(sinkNode);
    // iterate graph, mark segments and clusters.
    while (!iterStack.empty()) {
        const Node* node = iterStack.top();
        iterStack.pop();
        int id = node->id();
        if (id2Visit[id])
            continue;

        bool isCandidate = id2Candidate[id];
        // Root of a new segment.
        if (isCandidate && (id2Segment.find(id) == id2Segment.end()))
            id2Segment[id] = maxSegmentNum++;
        id2Visit[id] = true;

        std::unordered_map<int, bool> id2Enqueue;
        for (const Edge* edge : node->in_edges()) {
            bool isCtrlEdge = edge->IsControlEdge();
            Node* nextNode = edge->src();
            int nextId = nextNode->id();
            // Track unique data inputs..            
            if (id2Enqueue.find(nextId) != id2Enqueue.end())
                continue;
            if (id2Order[nextId] >= id2Order[id]) {
                // TODO: Encounter a cycle, might need to detect cycle ahead
                // of time.
                CHECK(false) << "TODO: encounter a circle";
            }
            if (!isCtrlEdge)
                id2Enqueue[nextId] = true;
            bool bothAreCandidates = (isCandidate && id2Candidate[nextId]);
            if (id2Visit.find(nextId) == id2Visit.end()) {
                if (bothAreCandidates && !isCtrlEdge)
                    id2Segment[nextId] = id2Segment[id];
                iterStack.push(nextNode);
            } else if (bothAreCandidates && !isCtrlEdge) {
#if 1                
                string name1 = node->def().name();
                string name2 = nextNode->def().name();
#endif
                // hash two segments into the same cluster.
                int nextSegmentId = id2Segment[nextId];
                if (segment2Cluster.find(nextSegmentId) == segment2Cluster.end())
                    segment2Cluster[nextSegmentId] = maxClusterNum++;
                segment2Cluster[id2Segment[id]] = segment2Cluster[nextSegmentId];
            }

            if (isCandidate && (!id2Candidate[nextId] || isCtrlEdge))
                id2Mask[id] |= (1 << is_entry);
            if ((!isCandidate || isCtrlEdge) && id2Candidate[nextId])
                id2Mask[nextId] |= (1 << is_exit);
        }
    }
    // Assign stand-alone segments to clusters.
    for (unsigned segmentId = 0; segmentId < maxSegmentNum; segmentId++) {
        if (segment2Cluster.find(segmentId) == segment2Cluster.end()) {
            segment2Cluster[segmentId] = maxClusterNum++;
        }
    }

    auto getClusterId = [&] (int nodeId)-> unsigned {
        unsigned segmentId = id2Segment[nodeId];
        unsigned clusterId = segment2Cluster[segmentId];
        return clusterId;
    };

    auto inCluster = [&] (int nodeId)-> bool {
        return  (id2Segment.find(nodeId) != id2Segment.end());
    };
    
    // Build clusters.
    if (maxClusterNum > 0) {
        std::vector<Cluster> clusters;;
        clusters.resize(maxClusterNum);
        for (Node* node : rpOrder) {
            int id = node->id();
            if (!inCluster(id))
                continue;
            unsigned clusterId = getClusterId(id);
            clusters[clusterId].addNode(node);
            if (id2Mask[id] & (1 << is_entry)) {
                for (const Edge* edge : node->in_edges()) {
                    Node* srcNode = edge->src();
                    int srcId = srcNode->id();
                    if (!inCluster(srcId) ||  (getClusterId(srcId) != clusterId)) {
                        clusters[clusterId].addInputEdge(edge);
                    }
                }
            }
            if (id2Mask[id] & (1 << is_exit)) {
                for (const Edge* edge : node->out_edges()) {
                    Node* dstNode = edge->dst();
                    int dstId = dstNode->id();
                    if (!inCluster(dstId) ||  (getClusterId(dstId) != clusterId)) {
                        clusters[clusterId].addOutputEdge(edge);
                    }
                }
            }
        }

        for (unsigned id = 0; id < maxClusterNum; id++) {
            Cluster& cluster = clusters[id];
            if (cluster.getSize() < MIN_CLUSTER_SIZE)
                continue;
            if ((cluster_dbg_limit >= 0) && (id > cluster_dbg_limit))
                continue;
            ConvertSubgraphToRTG(g, cluster, inputs, id2Mask, use_gpu, refiner);
        }
    }

    RTGLIB::dump_graph::DumpGraphToFile("After convert graph to RTG", graph);
    return Status::OK();
}
    
} // namspace convert
} // namespace rtglib
} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
