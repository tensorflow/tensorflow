/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"

#include "tensorflow/compiler/xla/layout_util.h"

namespace xla {
namespace gpu {

using se::dnn::DataLayout;
using se::dnn::DataLayoutString;
using se::dnn::FilterLayout;
using se::dnn::FilterLayoutString;

bool IsVoltaOrLater(const se::StreamExecutor& stream_executor) {
  int major, minor;
  CHECK(stream_executor.GetDeviceDescription().cuda_compute_capability(&major,
                                                                       &minor));
  return major >= 7;
}

StatusOr<std::tuple<Layout, Layout, Layout>>
StreamExecutorConvLayoutsToXlaLayouts(const ConvolutionDimensionNumbers& dnums,
                                      DataLayout input, FilterLayout filter,
                                      DataLayout output) {
  std::vector<int64> input_layout;
  switch (input) {
    case DataLayout::kBatchDepthYX:
      input_layout.push_back(dnums.input_batch_dimension());
      input_layout.push_back(dnums.input_feature_dimension());
      input_layout.insert(input_layout.end(),
                          dnums.input_spatial_dimensions().begin(),
                          dnums.input_spatial_dimensions().end());
      break;
    case DataLayout::kBatchYXDepth:
      input_layout.push_back(dnums.input_batch_dimension());
      input_layout.insert(input_layout.end(),
                          dnums.input_spatial_dimensions().begin(),
                          dnums.input_spatial_dimensions().end());
      input_layout.push_back(dnums.input_feature_dimension());
      break;
    default:
      return tensorflow::errors::Internal("Invalid input layout: ",
                                          DataLayoutString(input));
  }

  std::vector<int64> filter_layout;
  switch (filter) {
    case FilterLayout::kOutputInputYX:
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      break;
    case FilterLayout::kOutputYXInput:
      filter_layout.push_back(dnums.kernel_output_feature_dimension());
      filter_layout.insert(filter_layout.end(),
                           dnums.kernel_spatial_dimensions().begin(),
                           dnums.kernel_spatial_dimensions().end());
      filter_layout.push_back(dnums.kernel_input_feature_dimension());
      break;
    default:
      return tensorflow::errors::Internal("Invalid filter layout: ",
                                          FilterLayoutString(filter));
  }

  std::vector<int64> output_layout;
  switch (output) {
    case DataLayout::kBatchDepthYX:
      output_layout.push_back(dnums.output_batch_dimension());
      output_layout.push_back(dnums.output_feature_dimension());
      output_layout.insert(output_layout.end(),
                           dnums.output_spatial_dimensions().begin(),
                           dnums.output_spatial_dimensions().end());
      break;
    case DataLayout::kBatchYXDepth:
      output_layout.push_back(dnums.output_batch_dimension());
      output_layout.insert(output_layout.end(),
                           dnums.output_spatial_dimensions().begin(),
                           dnums.output_spatial_dimensions().end());
      output_layout.push_back(dnums.output_feature_dimension());
      break;
    default:
      return tensorflow::errors::Internal("Invalid output layout: ",
                                          DataLayoutString(output));
  }

  return std::make_tuple(LayoutUtil::MakeLayoutFromMajorToMinor(input_layout),
                         LayoutUtil::MakeLayoutFromMajorToMinor(filter_layout),
                         LayoutUtil::MakeLayoutFromMajorToMinor(output_layout));
}

StatusOr<std::tuple<DataLayout, FilterLayout, DataLayout>>
XlaConvLayoutsToStreamExecutorLayouts(const ConvolutionDimensionNumbers& dnums,
                                      const Layout& input, const Layout& filter,
                                      const Layout& output) {
  Layout nchw_input, nchw_filter, nchw_output;
  std::tie(nchw_input, nchw_filter, nchw_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchDepthYX,
                                            FilterLayout::kOutputInputYX,
                                            DataLayout::kBatchDepthYX)
          .ConsumeValueOrDie();

  Layout nhwc_input, nhwc_filter, nhwc_output;
  std::tie(nhwc_input, nhwc_filter, nhwc_output) =
      StreamExecutorConvLayoutsToXlaLayouts(dnums, DataLayout::kBatchYXDepth,
                                            FilterLayout::kOutputYXInput,
                                            DataLayout::kBatchYXDepth)
          .ConsumeValueOrDie();

  DataLayout input_layout;
  if (LayoutUtil::Equal(input, nchw_input)) {
    input_layout = DataLayout::kBatchDepthYX;
  } else if (LayoutUtil::Equal(input, nhwc_input)) {
    input_layout = DataLayout::kBatchYXDepth;
  } else {
    return tensorflow::errors::Internal("Invalid input layout: ",
                                        input.ShortDebugString());
  }

  FilterLayout filter_layout;
  if (LayoutUtil::Equal(filter, nchw_filter)) {
    filter_layout = FilterLayout::kOutputInputYX;
  } else if (LayoutUtil::Equal(filter, nhwc_filter)) {
    filter_layout = FilterLayout::kOutputYXInput;
  } else {
    return tensorflow::errors::Internal("Invalid filter layout: ",
                                        filter.ShortDebugString());
  }

  DataLayout output_layout;
  if (LayoutUtil::Equal(output, nchw_output)) {
    output_layout = DataLayout::kBatchDepthYX;
  } else if (LayoutUtil::Equal(output, nhwc_output)) {
    output_layout = DataLayout::kBatchYXDepth;
  } else {
    return tensorflow::errors::Internal("Invalid output layout: ",
                                        output.ShortDebugString());
  }

  return std::make_tuple(input_layout, filter_layout, output_layout);
}
}  // namespace gpu
}  // namespace xla
