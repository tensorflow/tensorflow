/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

%include "tensorflow/python/platform/base.i"

%typemap(in) const tensorflow::RunMetadata& (tensorflow::RunMetadata temp) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }

  if (!temp.ParseFromString(string(c_string, py_size))) {
    PyErr_SetString(
        PyExc_TypeError,
        "The MetaGraphDef could not be parsed as a valid protocol buffer");
    SWIG_fail;
  }
  $1 = &temp;
}

%typemap(in) const string& (string temp) {
  char *buf;
  Py_ssize_t len;
  if (PyBytes_AsStringAndSize($input, &buf, &len) == -1) return NULL;
  temp.assign(buf, len);
  $1 = &temp;
}

%{
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/costs/measuring_cost_estimator.h"
#include "tensorflow/core/grappler/costs/utils.h"

static tensorflow::grappler::Cluster* TF_NewCluster(
    bool allow_soft_placement, bool disable_detailed_stats, TF_Status* out_status) {
  int num_cpu_cores = tensorflow::grappler::GetNumAvailableLogicalCPUCores();
  int num_gpus = tensorflow::grappler::GetNumAvailableGPUs();;
  int timeout_s = 60 * 10;
  tensorflow::grappler::Cluster* cluster = new tensorflow::grappler::SingleMachine(
      timeout_s, num_cpu_cores, num_gpus);
  cluster->DisableDetailedStats(disable_detailed_stats);
  cluster->AllowSoftPlacement(allow_soft_placement);
  tensorflow::Status status = cluster->Provision();
  tensorflow::Set_TF_Status_from_Status(out_status, status);
  return cluster;
}

static void TF_DeleteCluster(tensorflow::grappler::Cluster* cluster) {
  cluster->Shutdown();
  delete cluster;
}

tensorflow::Status _GetOpPerformanceDataAndRunTime(const tensorflow::grappler::GrapplerItem& item,
                                       tensorflow::grappler::CostEstimator* cost_measure,
                                       tensorflow::OpPerformanceList* op_performance_data,
                                       tensorflow::grappler::Costs* costs) {
  tensorflow::Status status = cost_measure->Initialize(item);
  if (!status.ok()) return status;

  tensorflow::CostGraphDef cost_graph;
  TF_RETURN_IF_ERROR(
      cost_measure->PredictCosts(item.graph, &cost_graph, costs));

  if (op_performance_data) {
    *op_performance_data = tensorflow::grappler::CostGraphToOpPerformanceData(
        cost_graph, item.graph);
  }
  return tensorflow::Status::OK();
}

static PyObject* TF_MeasureCosts(
    const tensorflow::grappler::GrapplerItem* item, tensorflow::grappler::Cluster* cluster,
    bool generate_timeline, TF_Status* out_status) {
  tensorflow::OpPerformanceList op_performance_data;
  tensorflow::StepStats step_stats;

  tensorflow::grappler::MeasuringCostEstimator cost_measure(cluster, 10, 0);

  tensorflow::grappler::Costs costs;
  tensorflow::Status status = _GetOpPerformanceDataAndRunTime(*item, &cost_measure,
                                                 &op_performance_data, &costs);
  double run_time = FLT_MAX;
  if (status.ok()) {
    run_time = static_cast<double>(costs.execution_time.count()) / 1e9;
  }
  if (generate_timeline) {
    tensorflow::RunMetadata metadata;
    tensorflow::Status s = cluster->Run(item->graph, item->feed, item->fetch, &metadata);
    if (s.ok()) {
      step_stats = metadata.step_stats();
    } else {
      status = s;
    }
  }

  tensorflow::Set_TF_Status_from_Status(out_status, status);
  if (!status.ok()) {
    Py_RETURN_NONE;
  }
  PyObject* op_perf_objs = PyList_New(op_performance_data.op_performance_size());
  for (int i = 0; i < op_performance_data.op_performance_size(); i++) {
    string op_perf_str = op_performance_data.op_performance(i).SerializeAsString();
    PyObject* op_perf_obj = PyBytes_FromStringAndSize(op_perf_str.data(),
                                                      op_perf_str.size());
    PyList_SetItem(op_perf_objs, i, op_perf_obj);
  }

  PyObject* run_time_obj = PyFloat_FromDouble(run_time);

  string step_stats_str = step_stats.SerializeAsString();
  PyObject* metadata_obj = PyBytes_FromStringAndSize(step_stats_str.data(),
                                                     step_stats_str.size());

  PyObject* ret = PyTuple_New(3);
  if (PyTuple_SetItem(ret, 0, op_perf_objs) != 0 ||
      PyTuple_SetItem(ret, 1, run_time_obj) != 0 ||
      PyTuple_SetItem(ret, 2, metadata_obj) != 0) {
    Py_DECREF(ret);
    Py_XDECREF(op_perf_objs);
    Py_XDECREF(run_time_obj);
    Py_XDECREF(metadata_obj);
    status = tensorflow::Status(tensorflow::error::Code::INTERNAL,
                                "Error setting return tuples.");
    tensorflow::Set_TF_Status_from_Status(out_status, status);
    Py_RETURN_NONE;
  }
  return ret;
}

%}

// Wrap these functions.

static tensorflow::grappler::Cluster* TF_NewCluster(
    bool allow_soft_placement, bool disable_detailed_stats, TF_Status* out_status);
static void TF_DeleteCluster(tensorflow::grappler::Cluster* cluster);
static PyObject* TF_MeasureCosts(
    const tensorflow::grappler::GrapplerItem* item, tensorflow::grappler::Cluster* cluster,
    bool generate_timeline, TF_Status* out_status);
