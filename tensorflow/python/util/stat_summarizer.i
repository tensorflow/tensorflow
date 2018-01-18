/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

%include <std_string.i>
%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "tensorflow/python/lib/core/py_func.h"

#include "tensorflow/core/framework/step_stats.pb.h"
%}

%ignoreall

%unignore _NewStatSummarizer;
%unignore _DeleteStatSummarizer;
%unignore tensorflow;
%unignore tensorflow::StatSummarizer;
%unignore tensorflow::StatSummarizer::StatSummarizer;
%unignore tensorflow::StatSummarizer::~StatSummarizer;
%unignore tensorflow::StatSummarizer::Initialize;
%unignore tensorflow::StatSummarizer::InitializeStr;
%unignore tensorflow::StatSummarizer::ProcessStepStats;
%unignore tensorflow::StatSummarizer::ProcessStepStatsStr;
%unignore tensorflow::StatSummarizer::PrintStepStats;
%unignore tensorflow::StatSummarizer::GetOutputString;


// TODO(ashankar): Remove the unused argument from the API.
%{
tensorflow::StatSummarizer* _NewStatSummarizer(
      const string& unused) {
  return new tensorflow::StatSummarizer(tensorflow::StatSummarizerOptions());
}
%}

%{
void _DeleteStatSummarizer(tensorflow::StatSummarizer* ss) {
  delete ss;
}
%}

tensorflow::StatSummarizer* _NewStatSummarizer(const string& unused);
void _DeleteStatSummarizer(tensorflow::StatSummarizer* ss);

%extend tensorflow::StatSummarizer {
  void ProcessStepStatsStr(const string& step_stats_str) {
    tensorflow::StepStats step_stats;
    step_stats.ParseFromString(step_stats_str);
    $self->ProcessStepStats(step_stats);
}
}

%extend tensorflow::StatSummarizer {
  StatSummarizer() {
    tensorflow::StatSummarizer* ss = new tensorflow::StatSummarizer(
      tensorflow::StatSummarizerOptions());
    return ss;
}
}

%include "tensorflow/core/util/stat_summarizer.h"
%unignoreall

%insert("python") %{

# Wrapping NewStatSummarizer and DeletStatSummarizer because
# SWIG-generated functions are built-in functions and do not support
# setting _tf_api_names attribute.

def NewStatSummarizer(unused):
  return _NewStatSummarizer(unused)

def DeleteStatSummarizer(stat_summarizer):
  _DeleteStatSummarizer(stat_summarizer)

NewStatSummarizer._tf_api_names = ["contrib.stat_summarizer.NewStatSummarizer"]
DeleteStatSummarizer._tf_api_names = [
    "contrib.stat_summarizer.DeleteStatSummarizer"]
StatSummarizer._tf_api_names = ["contrib.stat_summarizer.StatSummarizer"]
%}
