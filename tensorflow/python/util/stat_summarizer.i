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

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
%}

%ignoreall

%unignore NewStatSummarizer;
%unignore DeleteStatSummarizer;
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


%{
tensorflow::StatSummarizer* NewStatSummarizer(
      const string& graph_def_str) {
  tensorflow::GraphDef graph_def;
  graph_def.ParseFromString(graph_def_str);
  return new tensorflow::StatSummarizer(graph_def);
}
%}


%{
void DeleteStatSummarizer(tensorflow::StatSummarizer* ss) {
  delete ss;
}
%}

tensorflow::StatSummarizer* NewStatSummarizer(const string& graph_def_str);
void DeleteStatSummarizer(tensorflow::StatSummarizer* ss);

%extend tensorflow::StatSummarizer {
  void ProcessStepStatsStr(const string& step_stats_str) {
    tensorflow::StepStats step_stats;
    step_stats.ParseFromString(step_stats_str);
    $self->ProcessStepStats(step_stats);
}
}

%extend tensorflow::StatSummarizer {
  StatSummarizer(const string& graph_def_str) {
    tensorflow::GraphDef graph_def;
    graph_def.ParseFromString(graph_def_str);
    tensorflow::StatSummarizer* ss = new tensorflow::StatSummarizer(graph_def);
    return ss;
}
}

%include "tensorflow/core/util/stat_summarizer.h"
%unignoreall
