#include "tensorflow/compiler/tf2tensorrt/convert/fixtures/op_converter_fixture.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/core/platform/status_matchers.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {
using ::tensorflow::testing::IsOk;
using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;

void OpConverterTest::Reset(TrtPrecisionMode precision_mode_to_test,
                            TrtTestMode trt_mode) {
  // Destroy existing TRT objects in a proper order.
  converter_.reset(nullptr);
  engine_.reset(nullptr);

  // Re-create them in proper order.
  converter_ = std::move(Converter::Create(precision_mode_to_test,
                                           /*use_calibration=*/false, &logger_,
                                           /*use_implicit_batch=*/trt_mode ==
                                               TrtTestMode::kImplicitBatch,
                                           /*engine_name=*/"")
                             .ValueOrDie());

  // Reset other related artifacts.
  scope_ = Scope::NewRootScope();
}

void OpConverterTest::CheckDataTypeMatches(const DataVec& datas) {
  if (VLOG_IS_ON(2)) {
    int nbBindings = engine_->getNbBindings();
    VLOG(2) << "Number of engine bindings: " << nbBindings;
    for (int i = 0; i < nbBindings; i++) {
      VLOG(2) << "Binding " << i << " name: " << engine_->getBindingName(i);
    }
  }
  for (const auto& data : datas) {
    VLOG(2) << "Checking if data type matches for tensor " << data.name;
    const int input_index = engine_->getBindingIndex(data.name.c_str());
    ASSERT_NE(-1, input_index);
    const nvinfer1::DataType trt_dtype =
        engine_->getBindingDataType(input_index);
    DataType tf_type;
    TF_ASSERT_OK(TrtTypeToTfType(trt_dtype, &tf_type));
    ASSERT_EQ(data.tensor.dtype(), tf_type)
        << DataTypeString(data.tensor.dtype()) << " vs. "
        << DataTypeString(tf_type);
  }
}

Status OpConverterTest::BuildAndRun(const DataVec& input_data,
                                    DataVec* output_data,
                                    const int batch_size) {
  // Mark the output tensor as TRT engine output.
  std::vector<Converter::EngineOutputInfo> output_info;
  for (const auto& data : *output_data) {
    nvinfer1::DataType trt_type;
    TF_RETURN_IF_ERROR(TfTypeToTrtType(data.tensor.dtype(), &trt_type));
    output_info.push_back({data.name, data.name, trt_type});
  }
  TF_RETURN_IF_ERROR(converter_->RenameAndMarkOutputTensors(output_info));

  // Build the TRT engine.
  if (engine_.get() != nullptr) {
    return errors::Internal("Engine already exists");
  }
  TrtShapeOptimizationProfile profiles;
  if (!converter_->use_implicit_batch()) {
    profiles.SetShapeTensorMask(converter_->network());
    TF_RETURN_IF_ERROR(profiles.CollectShapeValues(input_data));
    // Create a single optimization profile for explicit batch mode
    std::vector<TensorShape> input_shapes;
    TF_RETURN_IF_ERROR(GetShapeFromDataVec(input_data, &input_shapes));
    profiles.AddShape(input_shapes);
    std::vector<PartialTensorShape> input_partial_shapes;
    TF_RETURN_IF_ERROR(
        GetNetworkInputShapes(converter_->network(), &input_partial_shapes));
    profiles.InitProfiles(input_partial_shapes,
                          ProfileStrategy::kImplicitBatchModeCompatible);
  }
  TF_RETURN_IF_ERROR(
      converter_->BuildCudaEngine(&engine_,
                                  /*max_batch_size=*/batch_size,
                                  /*max_workspace_size_bytes=*/1 << 26,
                                  /*allocator=*/nullptr,
                                  /*calibrator=*/nullptr,
                                  /*profiles=*/&profiles));
  CHECK_NOTNULL(engine_.get());
  CheckDataTypeMatches(input_data);
  CheckDataTypeMatches(*output_data);

  const int num_bindings = input_data.size() + output_data->size();
  std::vector<void*> buffers(num_bindings);

  if (engine_->getNbBindings() != num_bindings) {
    return errors::Internal("Number of bindings do not match");
  }
  // Since we have only 1 optimization profile (which is enabled by default)
  // it is fine to create execution context directly, instead of calling
  // profiles.CreateExecutionContexts()
  TrtUniquePtrType<nvinfer1::IExecutionContext> execution_context(
      engine_->createExecutionContext());

  // Prepare input bindings.
  TF_RETURN_IF_ERROR(
      SetTrtEngineInputs(engine_.get(), execution_context.get(), 0, buffers,
                         converter_->use_implicit_batch(), batch_size, profiles,
                         nullptr, &input_data));
  // Prepare output bindings.
  TF_RETURN_IF_ERROR(SetTrtEngineOutputs(
      engine_.get(), execution_context.get(), 0, buffers,
      converter_->use_implicit_batch(), batch_size, nullptr, output_data));
  // Execute the TRT engine.
  TF_RETURN_IF_ERROR(TrtEnqueue(execution_context.get(), buffers, stream_,
                                converter_->use_implicit_batch(), batch_size));
  cudaStreamSynchronize(stream_);
  return Status::OK();
}

void OpConverterTest::AddTestTensorWithTFDims(const string& name,
                                              const std::vector<int32>& dims,
                                              nvinfer1::DataType trt_type,
                                              Status add_input_status) {
  DataType tf_type;
  TF_ASSERT_OK(TrtTypeToTfType(trt_type, &tf_type));
  ops::Placeholder::Attrs attrs;
  TF_EXPECT_OK(TensorShapeUtils::MakeShape(dims, &attrs.shape_));

  auto input = ops::Placeholder(scope_.WithOpName(name), tf_type, attrs);
  node_inputs_[name] = input.output;

  // Add a real ITensor for conversion conditionally.
  nvinfer1::Dims trt_dims;
  Status status = TensorShapeToTrtDims(
      attrs.shape_, converter_->use_implicit_batch(), &trt_dims);
  if (converter_->use_implicit_batch() && !status.ok()) {
    ASSERT_EQ(add_input_status, status);
    return;
  } else {
    TF_EXPECT_OK(status);
  }
  if (!converter_->use_implicit_batch() || HasStaticShape(trt_dims)) {
    int batch_size = dims.size() > 0 ? dims[0] : 0;
    Status status =
        converter_->AddInputTensor(name, trt_type, trt_dims, batch_size);
    ASSERT_EQ(add_input_status, status);
  }
}

void OpConverterTest::AddTestTensor(const string& name,
                                    const std::vector<int32>& dims,
                                    int batch_size,
                                    nvinfer1::DataType trt_dtype) {
  std::vector<int32> dims_with_batch(dims.size() + 1);
  dims_with_batch[0] = batch_size;
  std::copy(dims.begin(), dims.end(), dims_with_batch.begin() + 1);
  AddTestTensorWithTFDims(name, dims_with_batch, trt_dtype);
  if (HasStaticShape(dims)) {
    ASSERT_EQ(batch_size, converter_->batch_size_);
  }
}

Status OpConverterTest::RunValidation(const Node* node) {
  grappler::GrapplerItem item;
  TF_EXPECT_OK(scope_.ToGraphDef(&item.graph));
  grappler::GraphProperties graph_properties(item);
  TF_EXPECT_OK(graph_properties.InferStatically(true));

  TrtNodeValidator validator(graph_properties, converter_->precision_mode(),
                             /*use_calibration=*/false,
                             converter_->use_implicit_batch());
  return validator.IsTensorRTCandidate(node);
}

void OpConverterTest::RunConversion(const Node* node, error::Code expected_code,
                                    const std::string& expected_msg_substr) {
  EXPECT_THAT(converter_->ConvertNode(node->def()),
              StatusIs(expected_code, HasSubstr(expected_msg_substr)));
  if (expected_code == error::OK) {
    EXPECT_THAT(converter_->network(), LayerNamesNonEmpty());
  }
}

void OpConverterTest::RunValidationAndConversion(
    const NodeDef& node_def, error::Code expected_code,
    const std::string& expected_msg_substr, bool should_run_conversion) {
  // Add the node to the graph.
  // TODO(laigd): we should accept a function that adds the node using
  // `scope_`, so individual test case can reuse the scope object and we
  // don't need to add the edges here by ourselves.
  Graph* graph = scope_.graph();
  Status status;
  Node* node = graph->AddNode(std::move(node_def), &status);
  TF_EXPECT_OK(status);
  for (int i = 0; i < node_def.input().size(); ++i) {
    const string& input_name = node_def.input(i);
    const auto& itr = node_inputs_.find(input_name);
    QCHECK(itr != node_inputs_.end());
    const Output& input = itr->second;
    graph->AddEdge(input.node(), input.index(), node, i);
  }

  status = RunValidation(node);
  if (should_run_conversion && status.ok()) {
    RunConversion(node, expected_code, expected_msg_substr);
  } else {
    ASSERT_THAT(status,
                StatusIs(expected_code, HasSubstr(expected_msg_substr)));
  }
}

void OpConverterTest::RunValidationAndConversion(
    const NodeDef& node_def, const Status& status,
    const std::string& output_name,
    const std::vector<std::vector<int>>& exp_out_dims) {
  RunValidationAndConversion(node_def, status.code(), status.error_message(),
                             true);

  if (status.ok()) {
    // TODO(tfeher): Enable this check in explicit_batch_mode.
    // In dynamic shape mode the output dims cannot be tested here. In that
    // case we need to wait for the concrate input shapes to be defined (by
    // setBindingDimensions before enqueue) before we can check the output
    // dims.
    if (converter_->use_implicit_batch()) {
      for (int i = 0; i < exp_out_dims.size(); i++) {
        TRT_TensorOrWeights output;
        string name = i == 0 ? output_name : StrCat(output_name, ":", i);
        TF_EXPECT_OK(GetTensorOrWeights(name.c_str(), &output));
        ASSERT_TRUE(output.is_tensor());
        if (!exp_out_dims[i].empty()) {
          // Removing batch dim.
          auto out_dims = std::vector<int>(exp_out_dims[i].begin() + 1,
                                           exp_out_dims[i].end());
          VLOG(2) << "Testing output shape for tensor " << name;
          EXPECT_THAT(output.tensor()->getDimensions(), DimsAreArray(out_dims));
        }
      }
    }
  }
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow