/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/support/codegen/android_java_generator.h"

#include <ctype.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/experimental/support/codegen/code_generator.h"
#include "tensorflow/lite/experimental/support/codegen/metadata_helper.h"
#include "tensorflow/lite/experimental/support/codegen/utils.h"
#include "tensorflow/lite/experimental/support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace support {
namespace codegen {

namespace {

using details_android_java::ModelInfo;
using details_android_java::TensorInfo;

// Helper class to organize the C++ code block as a generated code block.
// Using ctor and dtor to simulate an enter/exit schema like `with` in Python.
class AsBlock {
 public:
  AsBlock(CodeWriter* code_writer, const std::string& before,
          bool trailing_blank_line = false)
      : code_writer_(code_writer), trailing_blank_line_(trailing_blank_line) {
    code_writer_->AppendNoNewLine(before);
    code_writer_->Append(" {");
    code_writer_->Indent();
  }
  ~AsBlock() {
    code_writer_->Outdent();
    code_writer_->Append("}");
    if (trailing_blank_line_) {
      code_writer_->NewLine();
    }
  }

 private:
  CodeWriter* code_writer_;
  bool trailing_blank_line_;
};

// Declare the functions first, so that the functions can follow a logical
// order.
bool GenerateWrapperClass(CodeWriter*, const ModelInfo&, ErrorReporter*);
bool GenerateWrapperImports(CodeWriter*, const ModelInfo&, ErrorReporter*);
bool GenerateWrapperInputs(CodeWriter*, const ModelInfo&, ErrorReporter*);
bool GenerateWrapperOutputs(CodeWriter*, const ModelInfo&, ErrorReporter*);
bool GenerateWrapperMetadata(CodeWriter*, const ModelInfo&, ErrorReporter*);
bool GenerateWrapperAPI(CodeWriter*, const ModelInfo&, ErrorReporter*);

std::string GetModelVersionedName(const ModelMetadata* metadata) {
  std::string model_name = "MyModel";
  if (metadata->name() != nullptr && !(metadata->name()->str().empty())) {
    model_name = metadata->name()->str();
  }
  std::string model_version = "unknown";
  if (metadata->version() != nullptr && !(metadata->version()->str().empty())) {
    model_version = metadata->version()->str();
  }
  return model_name + " (Version: " + model_version + ")";
}

TensorInfo CreateTensorInfo(const TensorMetadata* metadata,
                            const std::string& name, bool is_input, int index,
                            ErrorReporter* err) {
  TensorInfo tensor_info;
  std::string tensor_identifier = is_input ? "input" : "output";
  tensor_identifier += " " + std::to_string(index);
  tensor_info.associated_axis_label_index = FindAssociatedFile(
      metadata, AssociatedFileType_TENSOR_AXIS_LABELS, tensor_identifier, err);
  tensor_info.associated_value_label_index = FindAssociatedFile(
      metadata, AssociatedFileType_TENSOR_VALUE_LABELS, tensor_identifier, err);
  if (is_input && (tensor_info.associated_axis_label_index >= 0 ||
                   tensor_info.associated_value_label_index >= 0)) {
    err->Warning(
        "Found label file on input tensor (%s). Label file for input "
        "tensor is not supported yet. The "
        "file will be ignored.",
        tensor_identifier.c_str());
  }
  if (tensor_info.associated_axis_label_index >= 0 &&
      tensor_info.associated_value_label_index >= 0) {
    err->Warning(
        "Found both axis label file and value label file for tensor (%s), "
        "which is not supported. Only the axis label file will be used.",
        tensor_identifier.c_str());
  }
  tensor_info.is_input = is_input;
  tensor_info.name = SnakeCaseToCamelCase(name);
  tensor_info.upper_camel_name = tensor_info.name;
  tensor_info.upper_camel_name[0] = toupper(tensor_info.upper_camel_name[0]);
  tensor_info.normalization_unit =
      FindNormalizationUnit(metadata, tensor_identifier, err);
  if (metadata->content()->content_properties_type() ==
      ContentProperties_ImageProperties) {
    if (metadata->content()
            ->content_properties_as_ImageProperties()
            ->color_space() == ColorSpaceType_RGB) {
      tensor_info.content_type = "image";
      tensor_info.wrapper_type = "TensorImage";
      tensor_info.processor_type = "ImageProcessor";
      return tensor_info;
    } else {
      err->Warning(
          "Found Non-RGB image on tensor (%s). Codegen currently does not "
          "support it, and regard it as a plain numeric tensor.",
          tensor_identifier.c_str());
    }
  }
  tensor_info.content_type = "tensor";
  tensor_info.wrapper_type = "TensorBuffer";
  tensor_info.processor_type = "TensorProcessor";
  return tensor_info;
}

ModelInfo CreateModelInfo(const ModelMetadata* metadata,
                          const std::string& package_name,
                          const std::string& model_class_name,
                          const std::string& model_asset_path,
                          ErrorReporter* err) {
  ModelInfo model_info;
  if (!CodeGenerator::VerifyMetadata(metadata, err)) {
    // TODO(b/150116380): Create dummy model info.
    err->Error("Validating metadata failed.");
    return model_info;
  }
  model_info.package_name = package_name;
  model_info.model_class_name = model_class_name;
  model_info.model_asset_path = model_asset_path;
  model_info.model_versioned_name = GetModelVersionedName(metadata);
  const auto* graph = metadata->subgraph_metadata()->Get(0);
  auto names = CodeGenerator::NameInputsAndOutputs(
      graph->input_tensor_metadata(), graph->output_tensor_metadata());
  std::vector<std::string> input_tensor_names = std::move(names.first);
  std::vector<std::string> output_tensor_names = std::move(names.second);
  for (int i = 0; i < graph->input_tensor_metadata()->size(); i++) {
    model_info.inputs.push_back(
        CreateTensorInfo(graph->input_tensor_metadata()->Get(i),
                         input_tensor_names[i], true, i, err));
  }
  for (int i = 0; i < graph->output_tensor_metadata()->size(); i++) {
    model_info.outputs.push_back(
        CreateTensorInfo(graph->output_tensor_metadata()->Get(i),
                         output_tensor_names[i], false, i, err));
  }
  return model_info;
}

void SetCodeWriterWithTensorInfo(CodeWriter* code_writer,
                                 const TensorInfo& tensor_info) {
  code_writer->SetTokenValue("NAME", tensor_info.name);
  code_writer->SetTokenValue("NAME_U", tensor_info.upper_camel_name);
  code_writer->SetTokenValue("CONTENT_TYPE", tensor_info.content_type);
  code_writer->SetTokenValue("WRAPPER_TYPE", tensor_info.wrapper_type);
  std::string wrapper_name = tensor_info.wrapper_type;
  wrapper_name[0] = tolower(wrapper_name[0]);
  code_writer->SetTokenValue("WRAPPER_NAME", wrapper_name);
  code_writer->SetTokenValue("PROCESSOR_TYPE", tensor_info.processor_type);
  code_writer->SetTokenValue("NORMALIZATION_UNIT",
                             std::to_string(tensor_info.normalization_unit));
  code_writer->SetTokenValue(
      "ASSOCIATED_AXIS_LABEL_INDEX",
      std::to_string(tensor_info.associated_axis_label_index));
  code_writer->SetTokenValue(
      "ASSOCIATED_VALUE_LABEL_INDEX",
      std::to_string(tensor_info.associated_value_label_index));
}

void SetCodeWriterWithModelInfo(CodeWriter* code_writer,
                                const ModelInfo& model_info) {
  code_writer->SetTokenValue("PACKAGE", model_info.package_name);
  code_writer->SetTokenValue("MODEL_PATH", model_info.model_asset_path);
  code_writer->SetTokenValue("MODEL_CLASS_NAME", model_info.model_class_name);
}

constexpr char JAVA_DEFAULT_PACKAGE[] = "default";

std::string ConvertPackageToPath(const std::string& package) {
  if (package == JAVA_DEFAULT_PACKAGE) {
    return "";
  }
  std::string path = package;
  std::replace(path.begin(), path.end(), '.', '/');
  return path;
}

bool IsImageUsed(const ModelInfo& model) {
  for (const auto& input : model.inputs) {
    if (input.content_type == "image") {
      return true;
    }
  }
  for (const auto& output : model.outputs) {
    if (output.content_type == "image") {
      return true;
    }
  }
  return false;
}

bool GenerateWrapperFileContent(CodeWriter* code_writer, const ModelInfo& model,
                                ErrorReporter* err) {
  code_writer->Append("// Generated by TFLite Support.");
  code_writer->Append("package {{PACKAGE}};");
  code_writer->NewLine();

  if (!GenerateWrapperImports(code_writer, model, err)) {
    err->Error("Fail to generate imports for wrapper class.");
    return false;
  }
  if (!GenerateWrapperClass(code_writer, model, err)) {
    err->Error("Fail to generate wrapper class.");
    return false;
  }
  code_writer->NewLine();
  return true;
}

bool GenerateWrapperImports(CodeWriter* code_writer, const ModelInfo& model,
                            ErrorReporter* err) {
  const std::string support_pkg = "org.tensorflow.lite.support.";
  std::vector<std::string> imports{
      "android.content.Context",
      "java.io.IOException",
      "java.nio.ByteBuffer",
      "java.nio.FloatBuffer",
      "java.util.Arrays",
      "java.util.HashMap",
      "java.util.List",
      "java.util.Map",
      "org.checkerframework.checker.nullness.qual.Nullable",
      "org.tensorflow.lite.DataType",
      "org.tensorflow.lite.Tensor.QuantizationParams",
      support_pkg + "common.FileUtil",
      support_pkg + "common.TensorProcessor",
      support_pkg + "common.ops.CastOp",
      support_pkg + "common.ops.DequantizeOp",
      support_pkg + "common.ops.NormalizeOp",
      support_pkg + "common.ops.QuantizeOp",
      support_pkg + "label.TensorLabel",
      support_pkg + "metadata.MetadataExtractor",
      support_pkg + "metadata.schema.NormalizationOptions",
      support_pkg + "model.Model",
      support_pkg + "model.Model.Device",
      support_pkg + "tensorbuffer.TensorBuffer",
  };
  if (IsImageUsed(model)) {
    for (const auto& target :
         {"image.ImageProcessor", "image.TensorImage", "image.ops.ResizeOp",
          "image.ops.ResizeOp.ResizeMethod"}) {
      imports.push_back(support_pkg + target);
    }
    imports.push_back("android.graphics.Bitmap");
  }

  std::sort(imports.begin(), imports.end());
  for (const auto target : imports) {
    code_writer->SetTokenValue("TARGET", target);
    code_writer->Append("import {{TARGET}};");
  }
  code_writer->NewLine();
  return true;
}

bool GenerateWrapperClass(CodeWriter* code_writer, const ModelInfo& model,
                          ErrorReporter* err) {
  code_writer->SetTokenValue("MODEL_VERSIONED_NAME",
                             model.model_versioned_name);
  code_writer->Append(
      R"(/** Wrapper class of model {{MODEL_VERSIONED_NAME}} */)");
  const auto code_block =
      AsBlock(code_writer, "public class {{MODEL_CLASS_NAME}}");
  code_writer->Append(R"(private final Metadata metadata;
private final Model model;
private static final String MODEL_NAME = "{{MODEL_PATH}}";)");
  for (const auto& tensor : model.outputs) {
    if (tensor.associated_axis_label_index >= 0) {
      code_writer->SetTokenValue("NAME", tensor.name);
      code_writer->Append("private final List<String> {{NAME}}Labels;");
    }
  }
  for (const auto& tensor : model.inputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append(
        "@Nullable private {{PROCESSOR_TYPE}} {{NAME}}Preprocessor;");
  }
  for (const auto& tensor : model.outputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append(
        "@Nullable private {{PROCESSOR_TYPE}} {{NAME}}Postprocessor;");
  }
  code_writer->NewLine();
  if (!GenerateWrapperInputs(code_writer, model, err)) {
    err->Error("Failed to generate input classes");
    return false;
  }
  code_writer->NewLine();
  if (!GenerateWrapperOutputs(code_writer, model, err)) {
    err->Error("Failed to generate output classes");
    return false;
  }
  code_writer->NewLine();
  if (!GenerateWrapperMetadata(code_writer, model, err)) {
    err->Error("Failed to generate the metadata class");
    return false;
  }
  code_writer->NewLine();
  if (!GenerateWrapperAPI(code_writer, model, err)) {
    err->Error("Failed to generate the common APIs");
    return false;
  }
  return true;
}

bool GenerateWrapperInputs(CodeWriter* code_writer, const ModelInfo& model,
                           ErrorReporter* err) {
  code_writer->Append("/** Input wrapper of {@link {{MODEL_CLASS_NAME}}} */");
  auto class_block = AsBlock(code_writer, "public class Inputs");
  for (const auto& tensor : model.inputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append("private {{WRAPPER_TYPE}} {{NAME}};");
  }
  code_writer->NewLine();
  // Ctor
  {
    auto ctor_block = AsBlock(code_writer, "public Inputs()");
    code_writer->Append(
        "Metadata metadata = {{MODEL_CLASS_NAME}}.this.metadata;");
    for (const auto& tensor : model.inputs) {
      SetCodeWriterWithTensorInfo(code_writer, tensor);
      if (tensor.content_type == "image") {
        code_writer->Append(
            "{{NAME}} = new TensorImage(metadata.get{{NAME_U}}Type());");
      } else {
        code_writer->Append(
            "{{NAME}} = "
            "TensorBuffer.createFixedSize(metadata.get{{NAME_U}}Shape(), "
            "metadata.get{{NAME_U}}Type());");
      }
    }
  }
  for (const auto& tensor : model.inputs) {
    code_writer->NewLine();
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    // Loaders
    if (tensor.content_type == "image") {
      {
        auto bitmap_loader_block =
            AsBlock(code_writer, "public void load{{NAME_U}}(Bitmap bitmap)");
        code_writer->Append(R"({{NAME}}.load(bitmap);
{{NAME}} = preprocess{{NAME_U}}({{NAME}});)");
      }
      code_writer->NewLine();
      {
        auto tensor_image_loader_block = AsBlock(
            code_writer, "public void load{{NAME_U}}(TensorImage tensorImage)");
        code_writer->Append("{{NAME}} = preprocess{{NAME_U}}(tensorImage);");
      }
    } else {  // content_type == "FEATURE" or "UNKNOWN"
      auto tensorbuffer_loader_block = AsBlock(
          code_writer, "public void load{{NAME_U}}(TensorBuffer tensorBuffer)");
      code_writer->Append("{{NAME}} = preprocess{{NAME_U}}(tensorBuffer);");
    }
    code_writer->NewLine();
    // Processor
    code_writer->Append(
        R"(private {{WRAPPER_TYPE}} preprocess{{NAME_U}}({{WRAPPER_TYPE}} {{WRAPPER_NAME}}) {
  if ({{NAME}}Preprocessor == null) {
    return {{WRAPPER_NAME}};
  }
  return {{NAME}}Preprocessor.process({{WRAPPER_NAME}});
}
)");
  }
  {
    const auto get_buffer_block = AsBlock(code_writer, "Object[] getBuffer()");
    code_writer->AppendNoNewLine("return new Object[] {");
    for (const auto& tensor : model.inputs) {
      SetCodeWriterWithTensorInfo(code_writer, tensor);
      code_writer->AppendNoNewLine("{{NAME}}.getBuffer(), ");
    }
    code_writer->Backspace(2);
    code_writer->Append("};");
  }
  return true;
}

bool GenerateWrapperOutputs(CodeWriter* code_writer, const ModelInfo& model,
                            ErrorReporter* err) {
  code_writer->Append("/** Output wrapper of {@link {{MODEL_CLASS_NAME}}} */");
  auto class_block = AsBlock(code_writer, "public class Outputs");
  for (const auto& tensor : model.outputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append("private final {{WRAPPER_TYPE}} {{NAME}};");
  }
  code_writer->NewLine();
  {
    const auto ctor_block = AsBlock(code_writer, "public Outputs()");
    code_writer->Append(
        "Metadata metadata = {{MODEL_CLASS_NAME}}.this.metadata;");
    for (const auto& tensor : model.outputs) {
      SetCodeWriterWithTensorInfo(code_writer, tensor);
      if (tensor.content_type == "image") {
        code_writer->Append(
            R"({{NAME}} = new TensorImage(metadata.get{{NAME_U}}Type());
{{NAME}}.load(TensorBuffer.createFixedSize(metadata.get{{NAME_U}}Shape(), metadata.get{{NAME_U}}Type()));)");
      } else {  // FEATURE, UNKNOWN
        code_writer->Append(
            "{{NAME}} = "
            "TensorBuffer.createFixedSize(metadata.get{{NAME_U}}Shape(), "
            "metadata.get{{NAME_U}}Type());");
      }
    }
  }
  for (const auto& tensor : model.outputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->NewLine();
    if (tensor.associated_axis_label_index >= 0) {
      if (tensor.content_type == "image") {
        err->Warning(
            "Axis label for images is not supported. The labels will "
            "be ignored.");
      } else {
        code_writer->Append(R"(public Map<String, Float> get{{NAME_U}}() {
  return new TensorLabel({{NAME}}Labels, postprocess{{NAME_U}}({{NAME}})).getMapWithFloatValue();
})");
      }
    } else {
      code_writer->Append(R"(public {{WRAPPER_TYPE}} get{{NAME_U}}() {
  return postprocess{{NAME_U}}({{NAME}});
})");
    }
    code_writer->NewLine();
    {
      auto processor_block =
          AsBlock(code_writer,
                  "private {{WRAPPER_TYPE}} "
                  "postprocess{{NAME_U}}({{WRAPPER_TYPE}} {{WRAPPER_NAME}})");
      code_writer->Append(R"(if ({{NAME}}Postprocessor == null) {
  return {{WRAPPER_NAME}};
}
return {{NAME}}Postprocessor.process({{WRAPPER_NAME}});)");
    }
  }
  code_writer->NewLine();
  {
    const auto get_buffer_block =
        AsBlock(code_writer, "Map<Integer, Object> getBuffer()");
    code_writer->Append("Map<Integer, Object> outputs = new HashMap<>();");
    for (int i = 0; i < model.outputs.size(); i++) {
      SetCodeWriterWithTensorInfo(code_writer, model.outputs[i]);
      code_writer->SetTokenValue("ID", std::to_string(i));
      code_writer->Append("outputs.put({{ID}}, {{NAME}}.getBuffer());");
    }
    code_writer->Append("return outputs;");
  }
  return true;
}

bool GenerateWrapperMetadata(CodeWriter* code_writer, const ModelInfo& model,
                             ErrorReporter* err) {
  code_writer->Append(
      "/** Metadata accessors of {@link {{MODEL_CLASS_NAME}}} */");
  const auto class_block = AsBlock(code_writer, "public static class Metadata");
  for (const auto& tensor : model.inputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append(R"(private final int[] {{NAME}}Shape;
private final DataType {{NAME}}DataType;
private final QuantizationParams {{NAME}}QuantizationParams;)");
    if (tensor.normalization_unit >= 0) {
      code_writer->Append(R"(private final float[] {{NAME}}Mean;
private final float[] {{NAME}}Stddev;)");
    }
  }
  for (const auto& tensor : model.outputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append(R"(private final int[] {{NAME}}Shape;
private final DataType {{NAME}}DataType;
private final QuantizationParams {{NAME}}QuantizationParams;)");
    if (tensor.normalization_unit >= 0) {
      code_writer->Append(R"(private final float[] {{NAME}}Mean;
private final float[] {{NAME}}Stddev;)");
    }
    if (tensor.associated_axis_label_index >= 0 ||
        tensor.associated_value_label_index >= 0) {
      code_writer->Append("private final List<String> {{NAME}}Labels;");
    }
  }
  code_writer->NewLine();
  {
    const auto ctor_block = AsBlock(
        code_writer,
        "public Metadata(ByteBuffer buffer, Model model) throws IOException");
    code_writer->Append(
        "MetadataExtractor extractor = new MetadataExtractor(buffer);");
    for (int i = 0; i < model.inputs.size(); i++) {
      SetCodeWriterWithTensorInfo(code_writer, model.inputs[i]);
      code_writer->SetTokenValue("ID", std::to_string(i));
      code_writer->Append(
          R"({{NAME}}Shape = extractor.getInputTensorShape({{ID}});
{{NAME}}DataType = extractor.getInputTensorType({{ID}});
{{NAME}}QuantizationParams = extractor.getInputTensorQuantizationParams({{ID}});)");
      if (model.inputs[i].normalization_unit >= 0) {
        code_writer->Append(
            R"(NormalizationOptions {{NAME}}NormalizationOptions =
    (NormalizationOptions) extractor.getInputTensorMetadata({{ID}}).processUnits({{NORMALIZATION_UNIT}}).options(new NormalizationOptions());
FloatBuffer {{NAME}}MeanBuffer = {{NAME}}NormalizationOptions.meanAsByteBuffer().asFloatBuffer();
{{NAME}}Mean = new float[{{NAME}}MeanBuffer.limit()];
{{NAME}}MeanBuffer.get({{NAME}}Mean);
FloatBuffer {{NAME}}StddevBuffer = {{NAME}}NormalizationOptions.stdAsByteBuffer().asFloatBuffer();
{{NAME}}Stddev = new float[{{NAME}}StddevBuffer.limit()];
{{NAME}}StddevBuffer.get({{NAME}}Stddev);)");
      }
    }
    for (int i = 0; i < model.outputs.size(); i++) {
      SetCodeWriterWithTensorInfo(code_writer, model.outputs[i]);
      code_writer->SetTokenValue("ID", std::to_string(i));
      code_writer->Append(
          R"({{NAME}}Shape = model.getOutputTensorShape({{ID}});
{{NAME}}DataType = extractor.getOutputTensorType({{ID}});
{{NAME}}QuantizationParams = extractor.getOutputTensorQuantizationParams({{ID}});)");
      if (model.outputs[i].normalization_unit >= 0) {
        code_writer->Append(
            R"(NormalizationOptions {{NAME}}NormalizationOptions =
    (NormalizationOptions) extractor.getInputTensorMetadata({{ID}}).processUnits({{NORMALIZATION_UNIT}}).options(new NormalizationOptions());
FloatBuffer {{NAME}}MeanBuffer = {{NAME}}NormalizationOptions.meanAsByteBuffer().asFloatBuffer();
{{NAME}}Mean = new float[{{NAME}}MeanBuffer.limit()];
{{NAME}}MeanBuffer.get({{NAME}}Mean);
FloatBuffer {{NAME}}StddevBuffer = {{NAME}}NormalizationOptions.stdAsByteBuffer().asFloatBuffer();
{{NAME}}Stddev = new float[{{NAME}}StddevBuffer.limit()];
{{NAME}}StddevBuffer.get({{NAME}}Stddev);)");
      }
      if (model.outputs[i].associated_axis_label_index >= 0) {
        code_writer->Append(R"(String {{NAME}}LabelsFileName =
    extractor.getOutputTensorMetadata({{ID}}).associatedFiles({{ASSOCIATED_AXIS_LABEL_INDEX}}).name();
{{NAME}}Labels = FileUtil.loadLabels(extractor.getAssociatedFile({{NAME}}LabelsFileName));)");
      } else if (model.outputs[i].associated_value_label_index >= 0) {
        code_writer->Append(R"(String {{NAME}}LabelsFileName =
    extractor.getOutputTensorMetadata({{ID}}).associatedFiles({{ASSOCIATED_VALUE_LABEL_INDEX}}).name();
{{NAME}}Labels = FileUtil.loadLabels(extractor.getAssociatedFile({{NAME}}LabelsFileName));)");
      }
    }
  }
  for (const auto& tensor : model.inputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append(R"(
public int[] get{{NAME_U}}Shape() {
  return Arrays.copyOf({{NAME}}Shape, {{NAME}}Shape.length);
}

public DataType get{{NAME_U}}Type() {
  return {{NAME}}DataType;
}

public QuantizationParams get{{NAME_U}}QuantizationParams() {
  return {{NAME}}QuantizationParams;
})");
    if (tensor.normalization_unit >= 0) {
      code_writer->Append(R"(
public float[] get{{NAME_U}}Mean() {
  return Arrays.copyOf({{NAME}}Mean, {{NAME}}Mean.length);
}

public float[] get{{NAME_U}}Stddev() {
  return Arrays.copyOf({{NAME}}Stddev, {{NAME}}Stddev.length);
})");
    }
  }
  for (const auto& tensor : model.outputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append(R"(
public int[] get{{NAME_U}}Shape() {
  return Arrays.copyOf({{NAME}}Shape, {{NAME}}Shape.length);
}

public DataType get{{NAME_U}}Type() {
  return {{NAME}}DataType;
}

public QuantizationParams get{{NAME_U}}QuantizationParams() {
  return {{NAME}}QuantizationParams;
})");
    if (tensor.normalization_unit >= 0) {
      code_writer->Append(R"(
public float[] get{{NAME_U}}Mean() {
  return Arrays.copyOf({{NAME}}Mean, {{NAME}}Mean.length);
}

public float[] get{{NAME_U}}Stddev() {
  return Arrays.copyOf({{NAME}}Stddev, {{NAME}}Stddev.length);
})");
    }
    if (tensor.associated_axis_label_index >= 0 ||
        tensor.associated_value_label_index >= 0) {
      code_writer->Append(R"(
public List<String> get{{NAME_U}}Labels() {
  return {{NAME}}Labels;
})");
    }
  }
  return true;
}

bool GenerateWrapperAPI(CodeWriter* code_writer, const ModelInfo& model,
                        ErrorReporter* err) {
  code_writer->Append(R"(public Metadata getMetadata() {
  return metadata;
}
)");
  code_writer->Append(R"(/**
 * Creates interpreter and loads associated files if needed.
 *
 * @throws IOException if an I/O error occurs when loading the tflite model.
 */
public {{MODEL_CLASS_NAME}}(Context context) throws IOException {
  this(context, MODEL_NAME, Device.CPU, 1);
}

/**
 * Creates interpreter and loads associated files if needed, but loading another model in the same
 * input / output structure with the original one.
 *
 * @throws IOException if an I/O error occurs when loading the tflite model.
 */
public {{MODEL_CLASS_NAME}}(Context context, String modelPath) throws IOException {
  this(context, modelPath, Device.CPU, 1);
}

/**
 * Creates interpreter and loads associated files if needed, with device and number of threads
 * configured.
 *
 * @throws IOException if an I/O error occurs when loading the tflite model.
 */
public {{MODEL_CLASS_NAME}}(Context context, Device device, int numThreads) throws IOException {
  this(context, MODEL_NAME, device, numThreads);
}

/**
 * Creates interpreter for a user-specified model.
 *
 * @throws IOException if an I/O error occurs when loading the tflite model.
 */
public {{MODEL_CLASS_NAME}}(Context context, String modelPath, Device device, int numThreads) throws IOException {
  model = new Model.Builder(context, modelPath).setDevice(device).setNumThreads(numThreads).build();
  metadata = new Metadata(model.getData(), model);)");
  for (const auto& tensor : model.inputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append(R"(
  {{PROCESSOR_TYPE}}.Builder {{NAME}}PreprocessorBuilder = new {{PROCESSOR_TYPE}}.Builder())");
    if (tensor.content_type == "image") {
      code_writer->Append(R"(      .add(new ResizeOp(
          metadata.get{{NAME_U}}Shape()[1],
          metadata.get{{NAME_U}}Shape()[2],
          ResizeMethod.NEAREST_NEIGHBOR)))");
    }
    if (tensor.normalization_unit >= 0) {
      code_writer->Append(
          R"(      .add(new NormalizeOp(metadata.get{{NAME_U}}Mean(), metadata.get{{NAME_U}}Stddev())))");
    }
    code_writer->Append(
        R"(      .add(new QuantizeOp(
          metadata.get{{NAME_U}}QuantizationParams().getZeroPoint(),
          metadata.get{{NAME_U}}QuantizationParams().getScale()))
      .add(new CastOp(metadata.get{{NAME_U}}Type()));
  {{NAME}}Preprocessor = {{NAME}}PreprocessorBuilder.build();)");
  }
  for (const auto& tensor : model.outputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->AppendNoNewLine(R"(
  {{PROCESSOR_TYPE}}.Builder {{NAME}}PostprocessorBuilder = new {{PROCESSOR_TYPE}}.Builder()
      .add(new DequantizeOp(
          metadata.get{{NAME_U}}QuantizationParams().getZeroPoint(),
          metadata.get{{NAME_U}}QuantizationParams().getScale())))");
    if (tensor.normalization_unit >= 0) {
      code_writer->AppendNoNewLine(R"(
      .add(new NormalizeOp(metadata.get{{NAME_U}}Mean(), metadata.get{{NAME_U}}Stddev())))");
    }
    code_writer->Append(R"(;
  {{NAME}}Postprocessor = {{NAME}}PostprocessorBuilder.build();)");
    if (tensor.associated_axis_label_index >= 0) {
      code_writer->Append(R"(
  {{NAME}}Labels = metadata.get{{NAME_U}}Labels();)");
    }
  }
  code_writer->Append("}");
  for (const auto& tensor : model.inputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append(R"(
public void reset{{NAME_U}}Preprocessor(@Nullable {{PROCESSOR_TYPE}} processor) {
  {{NAME}}Preprocessor = processor;
})");
  }
  for (const auto& tensor : model.outputs) {
    SetCodeWriterWithTensorInfo(code_writer, tensor);
    code_writer->Append(R"(
public void reset{{NAME_U}}Postprocessor(@Nullable {{PROCESSOR_TYPE}} processor) {
  {{NAME}}Postprocessor = processor;
})");
  }
  code_writer->Append(R"(
/** Creates inputs */
public Inputs createInputs() {
  return new Inputs();
}

/** Triggers the model. */
public Outputs run(Inputs inputs) {
  Outputs outputs = new Outputs();
  model.run(inputs.getBuffer(), outputs.getBuffer());
  return outputs;
}

/** Closes the model. */
public void close() {
  model.close();
})");
  return true;
}

bool GenerateBuildGradleContent(CodeWriter* code_writer,
                                const ModelInfo& model_info) {
  code_writer->Append(R"(buildscript {
    repositories {
        google()
        jcenter()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:3.2.1'
    }
}

allprojects {
    repositories {
        google()
        jcenter()
        flatDir {
            dirs 'libs'
        }
    }
}

apply plugin: 'com.android.library'

android {
    compileSdkVersion 29
    defaultConfig {
        targetSdkVersion 29
        versionCode 1
        versionName "1.0"
    }
    aaptOptions {
        noCompress "tflite"
    }
    compileOptions {
        sourceCompatibility = '1.8'
        targetCompatibility = '1.8'
    }
    lintOptions {
        abortOnError false
    }
}

configurations {
    libMetadata
}

dependencies {
    libMetadata 'org.tensorflow:tensorflow-lite-support:0.0.0-experimental-metadata-monolithic'
}

task downloadLibs(type: Sync) {
    from configurations.libMetadata
    into "$buildDir/libs"
    rename 'tensorflow-lite-support-0.0.0-experimental-metadata-monolithic.jar', "tensorflow-lite-support-metadata.jar"
}

preBuild.dependsOn downloadLibs

dependencies {
    compileOnly 'org.checkerframework:checker-qual:2.5.8'
    api 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
    api 'org.tensorflow:tensorflow-lite-support:0.0.0-nightly'
    api files("$buildDir/libs/tensorflow-lite-support-metadata.jar")
    implementation 'org.apache.commons:commons-compress:1.19'
})");
  return true;
}

bool GenerateAndroidManifestContent(CodeWriter* code_writer,
                                    const ModelInfo& model_info) {
  code_writer->Append(R"(<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="{{PACKAGE}}">
</manifest>)");
  return true;
}

bool GenerateDocContent(CodeWriter* code_writer, const ModelInfo& model_info) {
  code_writer->Append("# {{MODEL_CLASS_NAME}} Usage");
  code_writer->AppendNoNewLine(R"(
```
import {{PACKAGE}}.{{MODEL_CLASS_NAME}};

// 1. Initialize the Model
{{MODEL_CLASS_NAME}} model = null;

try {
    model = new {{MODEL_CLASS_NAME}}(context);  // android.content.Context
    // Create the input container.
    {{MODEL_CLASS_NAME}}.Inputs inputs = model.createInputs();
} catch (IOException e) {
    e.printStackTrace();
}

if (model != null) {

    // 2. Set the inputs)");
  for (const auto& t : model_info.inputs) {
    SetCodeWriterWithTensorInfo(code_writer, t);
    if (t.content_type == "image") {
      code_writer->Append(R"(
    // Load input tensor "{{NAME}}" from a Bitmap with ARGB_8888 format.
    Bitmap bitmap = ...;
    inputs.load{{NAME_U}}(bitmap);
    // Alternatively, load the input tensor "{{NAME}}" from a TensorImage.
    // Check out TensorImage documentation to load other image data structures.
    // TensorImage tensorImage = ...;
    // inputs.load{{NAME_U}}(tensorImage);)");
    } else {
      code_writer->Append(R"(
    // Load input tensor "{{NAME}}" from a TensorBuffer.
    // Check out TensorBuffer documentation to load other data structures.
    TensorBuffer tensorBuffer = ...;
    inputs.load{{NAME_U}}(tensorBuffer);)");
    }
  }
  code_writer->Append(R"(
    // 3. Run the model
    {{MODEL_CLASS_NAME}}.Outputs outputs = model.run(inputs);)");
  code_writer->Append(R"(
    // 4. Retrieve the results)");
  for (const auto& t : model_info.outputs) {
    SetCodeWriterWithTensorInfo(code_writer, t);
    if (t.associated_axis_label_index >= 0) {
      code_writer->SetTokenValue("WRAPPER_TYPE", "Map<String, Float>");
    }
    code_writer->Append(
        R"(    {{WRAPPER_TYPE}} {{NAME}} = outputs.get{{NAME_U}}();)");
  }
  code_writer->Append(R"(}
```)");
  return true;
}

GenerationResult::File GenerateWrapperFile(const std::string& module_root,
                                           const ModelInfo& model_info,
                                           ErrorReporter* err) {
  const auto java_path = JoinPath(module_root, "src/main/java");
  const auto package_path =
      JoinPath(java_path, ConvertPackageToPath(model_info.package_name));
  const auto file_path =
      JoinPath(package_path, model_info.model_class_name + JAVA_EXT);

  CodeWriter code_writer(err);
  code_writer.SetIndentString("  ");
  SetCodeWriterWithModelInfo(&code_writer, model_info);

  if (!GenerateWrapperFileContent(&code_writer, model_info, err)) {
    err->Error("Generating Java wrapper content failed.");
  }

  const auto java_file = code_writer.ToString();
  return GenerationResult::File{file_path, java_file};
}

GenerationResult::File GenerateBuildGradle(const std::string& module_root,
                                           const ModelInfo& model_info,
                                           ErrorReporter* err) {
  const auto file_path = JoinPath(module_root, "build.gradle");
  CodeWriter code_writer(err);
  SetCodeWriterWithModelInfo(&code_writer, model_info);
  if (!GenerateBuildGradleContent(&code_writer, model_info)) {
    err->Error("Generating build.gradle failed.");
  }
  const auto content = code_writer.ToString();
  return GenerationResult::File{file_path, content};
}

GenerationResult::File GenerateAndroidManifest(const std::string& module_root,
                                               const ModelInfo& model_info,
                                               ErrorReporter* err) {
  const auto file_path = JoinPath(module_root, "src/main/AndroidManifest.xml");
  CodeWriter code_writer(err);
  SetCodeWriterWithModelInfo(&code_writer, model_info);
  if (!GenerateAndroidManifestContent(&code_writer, model_info)) {
    err->Error("Generating AndroidManifest.xml failed.");
  }
  return GenerationResult::File{file_path, code_writer.ToString()};
}

GenerationResult::File GenerateDoc(const std::string& module_root,
                                   const ModelInfo& model_info,
                                   ErrorReporter* err) {
  std::string lower = model_info.model_class_name;
  for (int i = 0; i < lower.length(); i++) {
    lower[i] = std::tolower(lower[i]);
  }
  const auto file_path = JoinPath(module_root, lower + ".md");
  CodeWriter code_writer(err);
  SetCodeWriterWithModelInfo(&code_writer, model_info);
  if (!GenerateDocContent(&code_writer, model_info)) {
    err->Error("Generating doc failed.");
  }
  return GenerationResult::File{file_path, code_writer.ToString()};
}

}  // namespace

AndroidJavaGenerator::AndroidJavaGenerator(const std::string& module_root)
    : CodeGenerator(), module_root_(module_root) {}

GenerationResult AndroidJavaGenerator::Generate(
    const Model* model, const std::string& package_name,
    const std::string& model_class_name, const std::string& model_asset_path) {
  GenerationResult result;
  const ModelMetadata* metadata = GetMetadataFromModel(model);
  if (metadata == nullptr) {
    err_.Error(
        "Cannot find TFLite Metadata in the model. Codegen will generate "
        "nothing.");
    return result;
  }
  details_android_java::ModelInfo model_info = CreateModelInfo(
      metadata, package_name, model_class_name, model_asset_path, &err_);
  result.files.push_back(GenerateWrapperFile(module_root_, model_info, &err_));
  result.files.push_back(GenerateBuildGradle(module_root_, model_info, &err_));
  result.files.push_back(
      GenerateAndroidManifest(module_root_, model_info, &err_));
  result.files.push_back(GenerateDoc(module_root_, model_info, &err_));
  return result;
}

GenerationResult AndroidJavaGenerator::Generate(
    const char* model_storage, const std::string& package_name,
    const std::string& model_class_name, const std::string& model_asset_path) {
  const Model* model = GetModel(model_storage);
  return Generate(model, package_name, model_class_name, model_asset_path);
}

std::string AndroidJavaGenerator::GetErrorMessage() {
  return err_.GetMessage();
}

}  // namespace codegen
}  // namespace support
}  // namespace tflite
