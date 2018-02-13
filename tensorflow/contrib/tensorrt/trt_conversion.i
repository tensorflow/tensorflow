/*

  wrap trt_conversion

 */
%{
#define SWIG_FILE_WITH_INIT
%}
%include "std_string.i"
%include "std_pair.i"
%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"
%template(StringPair) std::pair<string,string>;
%template() std::pair<swig::SwigPtr_PyObject, swig::SwigPtr_PyObject>;

%{
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "tensorflow/contrib/tensorrt/convert/convert_graph.h"
%}

%ignoreall
%unignore tensorflow;
%unignore trt_convert;
%unignore calib_convert;

%{
std::pair<string,string> trt_convert(string graph_def_string,//const tensorflow::GraphDef&
                                     std::vector<string> output_names,
                                     size_t max_batch_size,
                                     size_t max_workspace_size_bytes,
                                     bool int8
    // unfortunately we can't use TF_Status here since it
    // is in c/c_api and brings in a lot of other libraries
    // which in turn declare ops. These ops are included
    // statically in our library and cause an abort when
    // module is loaded due to double registration
    // until Tensorflow properly exposes these headers
    // we have to work around this by returning a string
    // and converting it to exception on python side.
    //,TF_Status* out_status) {
) {
  string out_status;

  tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromString(graph_def_string)) {
    out_status="InvalidArgument;Couldn't interpret input as a GraphDef";
    return std::pair<string,string>{out_status,""};
  }

  if (!output_names.size()) {
    out_status="InvalidArgument;Size of the output_names vector is 0";
    return std::pair<string,string>{out_status,""};
    //return "";
  }
  tensorflow::GraphDef outGraph;
  tensorflow::Status conversion_status =
      tensorrt::convert::ConvertGraphDefToTensorRT(graph_def,
                                                   output_names,
                                                   max_batch_size,
                                                   max_workspace_size_bytes,
                                                   &outGraph,int8);
  if (!conversion_status.ok()) {
    auto retCode=(int)conversion_status.code();
    char buff[2000];
    snprintf(buff,2000,"%d;%s",retCode,conversion_status.error_message().c_str());
    out_status=buff;
    return std::pair<string,string>{out_status,""};
  }
  string result;
  if (!outGraph.SerializeToString(&result)) {
    out_status="InvalidArgument;Couldn't serialize output as a GraphDef";
    return std::pair<string,string>{out_status,""};
  }
  out_status="OK;All good!";
  return std::pair<string,string>{out_status,result};
}

std::pair<string,string> calib_convert(string graph_def_string  //  const tensorflow::GraphDef&
    // unfortunately we can't use TF_Status here since it
    // is in c/c_api and brings in a lot of other libraries
    // which in turn declare ops. These ops are included
    // statically in our library and cause an abort when
    // module is loaded due to double registration
    // until Tensorflow properly exposes these headers
    // we have to work around this by returning a string
    // and converting it to exception on python side.
    //,TF_Status* out_status) {
) {
  string out_status;

  tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromString(graph_def_string)) {
    out_status="InvalidArgument;Couldn't interpret input as a GraphDef";
    return std::pair<string,string>{out_status,""};
  }

  tensorflow::GraphDef outGraph;
  tensorflow::Status conversion_status =
      tensorrt::convert::ConvertCalibGraphToInferGraph(graph_def,
                                                   &outGraph);
  if (!conversion_status.ok()) {
    auto retCode=(int)conversion_status.code();
    char buff[2000];
    snprintf(buff,2000,"%d;%s",retCode,conversion_status.error_message().c_str());
    out_status=buff;
    return std::pair<string,string>{out_status,""};
  }
  string result;
  if (!outGraph.SerializeToString(&result)) {
    out_status="InvalidArgument;Couldn't serialize output as a GraphDef";
    return std::pair<string,string>{out_status,""};
  }
  out_status="OK;All good!";
  return std::pair<string,string>{out_status,result};
}
%}

std::pair<string,string> trt_convert(string graph_def_string,
				     std::vector<string> output_names,
				     size_t max_batch_size,
				     size_t max_workspace_size,bool int8);

std::pair<string,string> calib_convert(string graph_def_string);


%unignoreall
