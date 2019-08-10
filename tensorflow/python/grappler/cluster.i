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
%include <std_shared_ptr.i>
%include "item.i"

// Wrap the cluster into an object that swig can manipulate. This ensures it will call the object
// destructor upon garbage collection instead of leaking memory.
struct GCluster {
  std::shared_ptr<tensorflow::grappler::Cluster> cluster_;
};

%{
#include "tensorflow/core/protobuf/device_properties.pb.h"

template <>
bool _PyObjAs(PyObject *input, tensorflow::NamedDevice *out) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize(input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    return false;
  }

  tensorflow::NamedDevice named_device;
  if (!named_device.ParseFromString(string(c_string, py_size))) {
    PyErr_SetString(
        PyExc_TypeError,
        "The NamedDevice could not be parsed as a valid protocol buffer");
    return false;
  }
  if (out) *out = named_device;
  return true;
}
%}

%typemap(in) const std::vector<tensorflow::NamedDevice>& (std::vector<tensorflow::NamedDevice> temp) {
  if (!tf_vector_input_helper($input, &temp, &_PyObjAs<tensorflow::NamedDevice>)) {
    SWIG_fail;
  }
  $1 = &temp;
}

%typemap(in) const tensorflow::NamedDevice& (tensorflow::NamedDevice temp) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }

  if (!temp.ParseFromString(string(c_string, py_size))) {
    PyErr_SetString(
        PyExc_TypeError,
        "The NamedDevice could not be parsed as a valid protocol buffer");
    SWIG_fail;
  }
  $1 = &temp;
}

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
        "The RunMetadata could not be parsed as a valid protocol buffer");
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
#include <memory>
#include <vector>
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/graph_memory.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/costs/measuring_cost_estimator.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/memory_types.h"

// Provide the implementation of the GCluster struct here.
struct GCluster {
  GCluster() {}
  GCluster(tensorflow::grappler::Cluster* cluster) : cluster_(cluster) {}

  tensorflow::grappler::Cluster* operator->() const {
    return cluster_.get();
  }
  tensorflow::grappler::Cluster* get() const {
    return cluster_.get();
  }
  bool is_none() const {
    return cluster_.get() == nullptr;
  }

  std::shared_ptr<tensorflow::grappler::Cluster> cluster_;
};


static GCluster TF_NewCluster(bool allow_soft_placement,
                   bool disable_detailed_stats, TF_Status* status) {
  int num_cpu_cores = tensorflow::grappler::GetNumAvailableLogicalCPUCores();
  int num_gpus = tensorflow::grappler::GetNumAvailableGPUs();
  int timeout_s = 60 * 10;
  tensorflow::grappler::Cluster* cluster_ =
      new tensorflow::grappler::SingleMachine(
          timeout_s, num_cpu_cores, num_gpus);
  cluster_->DisableDetailedStats(disable_detailed_stats);
  cluster_->AllowSoftPlacement(allow_soft_placement);
  cluster_->SetNumWarmupSteps(10);
  tensorflow::Status s = cluster_->Provision();
  tensorflow::Set_TF_Status_from_Status(status, s);
  return GCluster(cluster_);
}

static GCluster TF_NewVirtualCluster(
    const std::vector<tensorflow::NamedDevice>& named_devices, TF_Status* status) {
  std::unordered_map<string, tensorflow::DeviceProperties> devices;
  for (const auto& named_device : named_devices) {
    devices[named_device.name()]= named_device.properties();
  }
  tensorflow::grappler::Cluster* cluster_ =
      new tensorflow::grappler::VirtualCluster(devices);
  PyGILState_STATE gstate = PyGILState_Ensure();
  tensorflow::Status s = cluster_->Provision();
  PyGILState_Release(gstate);
  tensorflow::Set_TF_Status_from_Status(status, s);
  return GCluster(cluster_);
}

static void TF_ShutdownCluster(GCluster cluster) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  cluster->Shutdown();
  PyGILState_Release(gstate);
}

tensorflow::Status _GetOpPerformanceDataAndRunTime(
    const tensorflow::grappler::GrapplerItem& item,
    tensorflow::grappler::CostEstimator* cost_measure,
    tensorflow::OpPerformanceList* op_performance_data,
    tensorflow::grappler::Costs* costs) {
  tensorflow::Status status = cost_measure->Initialize(item);
  if (!status.ok()) return status;

  tensorflow::RunMetadata run_metadata;
  TF_RETURN_IF_ERROR(
      cost_measure->PredictCosts(item.graph, &run_metadata, costs));

  if (op_performance_data) {
    *op_performance_data = tensorflow::grappler::CostGraphToOpPerformanceData(
        run_metadata.cost_graph(), item.graph);
  }
  return tensorflow::Status::OK();
}

static PyObject* TF_ListDevices(GCluster cluster) {
  const std::unordered_map<string, tensorflow::DeviceProperties>& devices = cluster->GetDevices();
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* result = PyList_New(devices.size());
  int i = 0;
  for (auto& dev : devices) {
    tensorflow::NamedDevice d;
    d.set_name(dev.first);
    *d.mutable_properties() = dev.second;
    string dev_str = d.SerializeAsString();
    PyObject* dev_obj = PyBytes_FromStringAndSize(dev_str.data(),
                                                  dev_str.size());
    PyList_SetItem(result, i, dev_obj);
    ++i;
  }
  PyGILState_Release(gstate);
  return result;
}

static PyObject* TF_ListAvailableOps() {
  tensorflow::OpRegistry* registry = tensorflow::OpRegistry::Global();
  std::vector<tensorflow::OpDef> ops;
  registry->GetRegisteredOps(&ops);
  std::vector<string> op_names;
  for (const tensorflow::OpDef& op : ops) {
    op_names.push_back(op.name());
  }
  std::sort(op_names.begin(), op_names.end());

  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* result = PyList_New(op_names.size());
  for (int i = 0; i < op_names.size(); ++i) {
    PyList_SetItem(result, i, PyString_FromString(op_names[i].c_str()));
  }
  PyGILState_Release(gstate);
  return result;
}

static PyObject* TF_GetSupportedDevices(GCluster cluster, GItem item) {
  if (cluster.is_none() || item.is_none()) {
    Py_RETURN_NONE;
  }
  const std::unordered_map<string, tensorflow::DeviceProperties>& devices = cluster->GetDevices();
  std::unordered_map<string, std::vector<string>> device_types;
  for (const auto& dev : devices) {
    device_types[dev.second.type()].push_back(dev.first);
  }

  std::unordered_map<string, std::set<string>> supported_device_types;
  std::unordered_map<string, std::set<string>> device_restrictions;

  for (const auto& node : item->graph.node()) {
    for (const auto& dev : device_types) {
      const string& type = dev.first;
      if (cluster->type() != "single_machine") {
        // The actual kernel may not be linked in this binary.
        supported_device_types[node.name()].insert(type);
      } else {
        // Check the kernel capabilities
        const tensorflow::DeviceType dev_type(type);
        tensorflow::Status s = tensorflow::FindKernelDef(dev_type, node, nullptr, nullptr);
        if (s.ok()) {
          supported_device_types[node.name()].insert(type);

          // Check which inputs are restricted to reside on the host.
          // TODO: extends this to support outputs as well
          tensorflow::MemoryTypeVector inp_mtypes;
          tensorflow::MemoryTypeVector out_mtypes;
          s = tensorflow::MemoryTypesForNode(tensorflow::OpRegistry::Global(), dev_type, node,
                                             &inp_mtypes, &out_mtypes);
          if (s.ok()) {
            for (int i = 0; i < inp_mtypes.size(); ++i) {
              if (inp_mtypes[i] == tensorflow::HOST_MEMORY) {
                device_restrictions[tensorflow::grappler::NodeName(node.input(i))].insert("CPU");
                break;
              }
            }
          }
        }
      }
    }
  }

  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* result = PyDict_New();

  for (const auto& supported_dev : supported_device_types) {
    const string& node = supported_dev.first;
    std::set<string> feasible;
    const auto it = device_restrictions.find(node);
    if (it != device_restrictions.end()) {
      const std::set<string>& candidates = supported_dev.second;
      const std::set<string>& valid = it->second;
      std::set_intersection(candidates.begin(), candidates.end(), valid.begin(), valid.end(),
                            std::inserter(feasible, feasible.begin()));
    } else {
      feasible = supported_dev.second;
    }

    std::vector<string> device_names;
    for (const string& type : feasible) {
      auto it = device_types.find(type);
      CHECK(it != device_types.end());
      for (const string& name : it->second) {
        device_names.push_back(name);
      }
    }

    PyObject* dev = PyList_New(device_names.size());
    for (int i = 0; i < device_names.size(); ++i) {
      PyList_SetItem(dev, i, PyString_FromString(device_names[i].c_str()));
    }
    CHECK_EQ(0, PyDict_SetItem(result, PyString_FromString(node.c_str()), dev));
  }
  PyGILState_Release(gstate);
  return result;
}


static double TF_EstimatePerformance(const tensorflow::NamedDevice& device) {
  tensorflow::grappler::OpLevelCostEstimator estimator;
  tensorflow::grappler::DeviceInfo info =
      estimator.GetDeviceInfo(device.properties());
  return info.gigaops;
}

static PyObject* TF_MeasureCosts(
    GItem item,
    GCluster cluster,
    bool generate_timeline, TF_Status* status) {
  tensorflow::OpPerformanceList op_performance_data;
  tensorflow::StepStats step_stats;

  const int num_measurements = cluster->type() == "virtual" ? 1 : 10;
  tensorflow::grappler::MeasuringCostEstimator cost_measure(cluster.get(), num_measurements, 0);

  tensorflow::grappler::Costs costs;
  tensorflow::Status s = _GetOpPerformanceDataAndRunTime(
      *item, &cost_measure, &op_performance_data, &costs);
  double run_time = FLT_MAX;
  if (s.ok()) {
    run_time = static_cast<double>(costs.execution_time.count()) / 1e9;
  }
  if (generate_timeline) {
    tensorflow::RunMetadata metadata;
    tensorflow::Status run_status = cluster->Run(
        item->graph, item->feed, item->fetch, &metadata);
    if (run_status.ok()) {
      step_stats = metadata.step_stats();
    } else {
      s = run_status;
    }
  }

  tensorflow::Set_TF_Status_from_Status(status, s);
  if (!s.ok()) {
    Py_RETURN_NONE;
  }
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* op_perf_objs = PyList_New(
      op_performance_data.op_performance_size());
  for (int i = 0; i < op_performance_data.op_performance_size(); i++) {
    string op_perf_str =
        op_performance_data.op_performance(i).SerializeAsString();
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
    s = tensorflow::Status(tensorflow::error::Code::INTERNAL,
                           "Error setting return tuples.");
    tensorflow::Set_TF_Status_from_Status(status, s);
    Py_INCREF(Py_None);
    ret = Py_None;
  }
  PyGILState_Release(gstate);
  return ret;
}


static PyObject* TF_DeterminePeakMemoryUsage(
    GItem item,
    GCluster cluster,
    TF_Status* status) {
  if (item.is_none() || cluster.is_none()) {
    tensorflow::Status s(tensorflow::error::Code::INTERNAL,
                         "You need both a cluster and an item to determine peak memory usage");
    tensorflow::Set_TF_Status_from_Status(status, s);
    Py_RETURN_NONE;
  }
  tensorflow::grappler::GraphMemory memory(*item);

  tensorflow::Status s;
  if (cluster->DetailedStatsEnabled()) {
    s = memory.InferDynamically(cluster.get());
  } else {
    s = memory.InferStatically(cluster->GetDevices());
  }
  if (!s.ok()) {
    tensorflow::Set_TF_Status_from_Status(status, s);
    Py_RETURN_NONE;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* result = PyDict_New();
  for (const auto& device : cluster->GetDevices()) {
    const tensorflow::grappler::GraphMemory::MemoryUsage& usage =
        memory.GetPeakMemoryUsage(device.first);
    PyObject* per_device = PyList_New(usage.live_tensors.size());
    for (int i = 0; i < usage.live_tensors.size(); ++i) {
      const auto& live_tensor = usage.live_tensors[i];
      PyObject* live = PyTuple_New(5);
      PyTuple_SetItem(live, 0, PyString_FromString(live_tensor.node.c_str()));
      PyTuple_SetItem(live, 1, PyInt_FromLong(live_tensor.output_id));
      PyTuple_SetItem(live, 2, PyLong_FromLong(live_tensor.memory_used));
      PyTuple_SetItem(live, 3, PyLong_FromLong(live_tensor.allocation_time.count()));
      PyTuple_SetItem(live, 4, PyLong_FromLong(live_tensor.deallocation_time.count()));
      PyList_SetItem(per_device, i, live);

    }
    PyObject* ret = PyTuple_New(2);
    PyTuple_SetItem(ret, 0, PyLong_FromLong(usage.used_memory));
    PyTuple_SetItem(ret, 1, per_device);
    PyDict_SetItem(result, PyString_FromString(device.first.c_str()), ret);
  }
  PyGILState_Release(gstate);
  return result;
}

%}

// Wrap these functions.
static GCluster TF_NewCluster(
    bool allow_soft_placement, bool disable_detailed_stats, TF_Status* status);
static GCluster TF_NewVirtualCluster(
    const std::vector<tensorflow::NamedDevice>& named_devices,
    TF_Status* status);
static void TF_ShutdownCluster(GCluster cluster);
static PyObject* TF_ListDevices(GCluster cluster);
static PyObject* TF_ListAvailableOps();
static PyObject* TF_GetSupportedDevices(GCluster cluster, GItem item);
static float TF_EstimatePerformance(const tensorflow::NamedDevice& device);
static PyObject* TF_MeasureCosts(
    GItem item, GCluster cluster,
    bool generate_timeline, TF_Status* status);
static PyObject* TF_DeterminePeakMemoryUsage(
    GItem item, GCluster cluster,
    TF_Status* status);
