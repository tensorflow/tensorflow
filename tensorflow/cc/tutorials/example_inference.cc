#include <unistd.h>

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

// Default batch size
#define BATCH_SIZE 16
// Default infer iterations for each thread
#define INFER_NUM 1000
// Default num of threads
#define NUM_THREADS 3

namespace tensorflow {

namespace example {

typedef std::vector<std::pair<std::string, Tensor>> InputsMap;

// thread function for multiple thread TF session run
void TFRun(Session *sess, int infer_num,
           std::vector<std::pair<std::string, Tensor>> *inputs,
           std::vector<std::string> *output_names,
           std::vector<Tensor> *output_tensors) {
  for (int i = 0; i < infer_num; i++) {
    TF_CHECK_OK(sess->Run(*inputs, *output_names, {}, output_tensors));
  }
}

// assume that the first unkonwn dim is batch size
TensorShape getNodeShape(const GraphDef &graph_def, const std::string name,
                         int batch_size) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    auto n = graph_def.node(i);
    if (n.name() == name) {
      auto shape = n.attr().at("shape").shape();
      int dims = shape.dim_size();
      TensorShape tensorShape;
      for (int d = 0; d < dims; d++) {
        int dim_size = shape.dim(d).size();
        if (d == 0 && dim_size == -1) {
          int new_size = batch_size;
          // assume the first dimension is batch size,
          // note that it may not be true for some models.
          // LOG(INFO) << "change batch size from: " << dim_size <<
          //    " to " << new_size << std::endl;
          dim_size = new_size;
        }
        tensorShape.AddDim(dim_size);
      }
      return tensorShape;
    }
  }
  LOG(ERROR) << "Cannot find the node" << name << std::endl;
  exit(1);
}

DataType getNodeType(const GraphDef &graph_def, const std::string name) {
  for (int i = 0; i < graph_def.node_size(); i++) {
    auto n = graph_def.node(i);
    if (n.name() == name) {
      auto dtype = n.attr().at("dtype").type();
      return dtype;
    }
  }
  LOG(ERROR) << "Cannot find the node" << name << std::endl;
  exit(1);
}

void RandomInitialize(Tensor &t) {
  int num_elements = t.NumElements();
  if (t.dtype() == DT_HALF) {
    auto *data = t.flat<Eigen::half>().data();
    for (int i = 0; i < num_elements; i++) {
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      data[i] = static_cast<Eigen::half>(value);
    }
  } else if (t.dtype() == DT_FLOAT) {
    float *data = t.flat<float>().data();
    for (int i = 0; i < num_elements; i++) {
      float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
      data[i] = value;
    }
  } else if (t.dtype() == DT_INT32) {
    int *data = t.flat<int>().data();
    for (int i = 0; i < num_elements; i++) {
      int value = static_cast<int>(rand() % 10);
      data[i] = value;
    }
  } else if (t.dtype() == DT_BOOL) {
    bool *data = t.flat<bool>().data();
    for (int i = 0; i < num_elements; i++) {
      bool value = static_cast<bool>(rand() % 2);
      data[i] = value;
    }
  } else if (t.dtype() == DT_INT64) {
    int64 *data = t.flat<int64>().data();
    for (int i = 0; i < num_elements; i++) {
      int64 value = static_cast<int64>(rand() % 10);
      data[i] = value;
    }
  } else {
    LOG(WARNING) << "Unsupported data type.";
  }
  return;
}

void PrintTensorData(Tensor &t) {
  void *data;
  if (t.dtype() == DT_HALF) {
    data = static_cast<void *>(t.flat<Eigen::half>().data());
  } else if (t.dtype() == DT_FLOAT) {
    data = static_cast<void *>(t.flat<float>().data());
  } else if (t.dtype() == DT_BOOL) {
    data = static_cast<void *>(t.flat<bool>().data());
  } else if (t.dtype() == DT_INT32) {
    data = static_cast<void *>(t.flat<int>().data());
  } else {
    LOG(WARNING) << "Print Tensor: Unsupported data type!";
    return;
  }

  int dims = t.dims();
  std::cout << "shape: " << std::endl;
  for (int i = 0; i < dims; i++) {
    std::cout << t.dim_size(i) << ", ";
  }
  std::cout << std::endl;

  // print the first 32 elements
  int size = t.NumElements();
  size = size > 32 ? 32 : size;

  for (int i = 0; i < size; i++) {
    float value;
    if (t.dtype() == DT_HALF) {
      value = static_cast<float>(static_cast<Eigen::half *>(data)[i]);
    } else if (t.dtype() == DT_INT32) {
      value = static_cast<int *>(data)[i];
    } else if (t.dtype() == DT_BOOL) {
      value = static_cast<bool *>(data)[i];
    } else {
      value = static_cast<float *>(data)[i];
    }
    std::cout << value << ", ";
  }
  std::cout << std::endl;
}

void GenerateInputs(GraphDef &graph_def, const std::vector<string> &input_names,
                    std::vector<Tensor> &input_tensors, int batch_size) {
  input_tensors.clear();
  for (int i = 0; i < input_names.size(); i++) {
    auto tensorshape = getNodeShape(graph_def, input_names[i], batch_size);
    auto tensortype = getNodeType(graph_def, input_names[i]);
    Tensor t = Tensor(tensortype, tensorshape);
    RandomInitialize(t);
    input_tensors.push_back(t);
  }
}

void FillInputsMap(InputsMap &inputs_map, std::vector<std::string> &input_names,
                   std::vector<Tensor> &input_tensors) {
  assert(input_names.size() == input_tensors.size());
  for (size_t i = 0; i < input_tensors.size(); i++) {
    inputs_map.push_back(
        std::pair<std::string, Tensor>(input_names[i], input_tensors[i]));
  }
}

void CopyTensorContents(Tensor &dst_tensor, Tensor &src_tensor) {
  int dst_eles = dst_tensor.NumElements();
  int src_eles = src_tensor.NumElements();
  if (dst_eles != src_eles) {
    LOG(ERROR) << "number of elements not match";
  }

  char *dst, *src;
  int ele_size;
  if (dst_tensor.dtype() == DT_HALF) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<Eigen::half>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<Eigen::half>().data());
    ele_size = 2;
  } else if (dst_tensor.dtype() == DT_FLOAT) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<float>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<float>().data());
    ele_size = 4;
  } else if (dst_tensor.dtype() == DT_INT32) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<int>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<int>().data());
    ele_size = 4;
  } else if (dst_tensor.dtype() == DT_BOOL) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<bool>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<bool>().data());
    ele_size = 1;
  } else if (dst_tensor.dtype() == DT_INT64) {
    dst = reinterpret_cast<char *>(dst_tensor.flat<int64>().data());
    src = reinterpret_cast<char *>(src_tensor.flat<int64>().data());
    ele_size = 8;
  } else {
    LOG(ERROR) << "Copy Tensor: Unsupported data type!" << std::endl;
    return;
  }

  for (int i = 0; i < src_eles * ele_size; i++) {
    dst[i] = src[i];
  }
}

inline void SetDevice(const string &device, GraphDef *graph_def) {
  for (int i = 0; i < graph_def->node_size(); ++i) {
    auto node = graph_def->mutable_node(i);
    if (node->device().empty()) {
      node->set_device(device);
    }
  }
}

Status Test(GraphDef &graph_def, std::vector<std::string> &input_names,
            std::vector<std::string> &output_names, int batch_size,
            int num_infers_per_thread, int num_threads, bool use_xla,
            int threadpool_num, int threadpool_size) {
  SessionOptions options;
  // options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  // options.config.mutable_gpu_options()->set_allow_growth(false);
  if (use_xla) {
    // enable XLA
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_global_jit_level(tensorflow::OptimizerOptions::ON_1);
    // These XLA flags are needed to trigger XLA properly from C (more
    // generally non-Python) clients. If this API is called again with
    // `enable` set to false, it is safe to keep these flag values as is.
    auto *flags = tensorflow::GetMarkForCompilationPassFlags();
    flags->tf_xla_cpu_global_jit = true;
    flags->tf_xla_min_cluster_size = 1;
    tensorflow::SetXlaAutoJitFlagFromFlagString("single-gpu(2)");
  }
  for (int i = 0; i < threadpool_num; ++i) {
    options.config.add_session_inter_op_thread_pool();
  }
  if (threadpool_size > 0) {
    options.config.set_inter_op_parallelism_threads(threadpool_size);
  }
  std::unique_ptr<Session> session(NewSession(options));

  // SetDevice("/device:GPU:0", &graph_def);
  TF_CHECK_OK(session->Create(graph_def));

  std::vector<Tensor> input_tensors_tf;
  GenerateInputs(graph_def, input_names, input_tensors_tf, batch_size);

  // first normal TF session run
  InputsMap inputs_tf;  // input map for Normal TF run
  FillInputsMap(inputs_tf, input_names, input_tensors_tf);

  std::vector<std::vector<Tensor>> output_tensors_tf(num_threads);
  std::vector<std::thread> threads;

  // init run
  // the first session run is slow due to resource initialization
  TFRun(session.get(), num_infers_per_thread, &inputs_tf, &output_names,
        &output_tensors_tf[0]);
  sleep(1);

  // the seconed session run can be used to compare single thread runing
  auto start = system_clock::now();
  TFRun(session.get(), num_infers_per_thread, &inputs_tf, &output_names,
        &output_tensors_tf[0]);
  auto end = system_clock::now();
  auto duration = duration_cast<milliseconds>(end - start);
  double duration_seconds = duration.count() * 1.0 / 1000;
  LOG(INFO) << "[TF] Single Thread Duration = " << duration_seconds
            << " seconds.";
  double qps = num_infers_per_thread * 1.0 / duration_seconds;
  LOG(INFO) << "[TF] Single Thread QPS = " << qps;
  sleep(1);

  start = system_clock::now();
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                  &inputs_tf, &output_names,
                                  &output_tensors_tf[i]));
  }

  for (auto &thread : threads) {
    thread.join();
  }
  end = system_clock::now();

  for (int i = 0; i < num_threads; i++) {
    LOG(INFO) << "TF results: ";
    // print first output tensor
    PrintTensorData(output_tensors_tf[i][0]);
    output_tensors_tf[i].clear();
  }

  duration = duration_cast<milliseconds>(end - start);
  duration_seconds = duration.count() * 1.0 / 1000;
  LOG(INFO) << "[TF] Multiple Threads Duration = " << duration_seconds
            << " seconds.";
  qps = num_infers_per_thread * num_threads * 1.0 / duration_seconds;
  LOG(INFO) << "[TF] Multiple Threads QPS = " << qps;

  TF_CHECK_OK(session->Close());
  return OkStatus();
}
}  // end namespace example
}  // end namespace tensorflow

using namespace tensorflow;

int main(int argc, char *argv[]) {
  // Example: ./application model_path in_num input_name [input_names]
  //                        out_num output_name [output_names]
  //                        batch_size infer_num_per_thread num_threads
  //                        use_xla number_of_threadpool
  //                        number_of_threads_in_one_threadpool

  int arg_idx = 1;
  std::string model_path = argv[arg_idx++];
  int input_num = std::stoi(argv[arg_idx++]);

  std::vector<std::string> input_names;
  std::cout << input_num << " inputs: ";
  for (int i = 0; i < input_num; i++) {
    input_names.push_back(argv[arg_idx++]);
    std::cout << argv[arg_idx - 1] << ",";
  }
  std::cout << std::endl;

  int output_num = std::stoi(argv[arg_idx++]);
  std::vector<std::string> output_names;
  std::cout << output_num << " outputs: ";
  for (int i = 0; i < output_num; i++) {
    output_names.push_back(argv[arg_idx++]);
    std::cout << argv[arg_idx - 1] << ",";
  }
  std::cout << std::endl;

  int batch_size = BATCH_SIZE;
  if (argc > arg_idx) {
    batch_size = std::stoi(argv[arg_idx++]);
  }
  std::cout << "batch size = " << batch_size << std::endl;

  int num_infers_per_thread = INFER_NUM;
  if (argc > arg_idx) {
    num_infers_per_thread = std::stoi(argv[arg_idx++]);
  }
  std::cout << "num_infers_per_thread = " << num_infers_per_thread << std::endl;

  int num_threads = NUM_THREADS;
  if (argc > arg_idx) {
    num_threads = std::stoi(argv[arg_idx++]);
  }
  std::cout << "num_threads = " << num_threads << std::endl;

  bool use_xla = false;
  if (argc > arg_idx) {
    use_xla = std::stoi(argv[arg_idx++]);
  }
  std::cout << "use_xla = " << use_xla << std::endl;

  int threadpool_num = 0;
  if (argc > arg_idx) {
    threadpool_num = std::stoi(argv[arg_idx++]);
  }
  std::cout << "threadpool_num = " << threadpool_num << std::endl;

  int threadpool_size = 0;
  if (argc > arg_idx) {
    threadpool_size = std::stoi(argv[arg_idx++]);
  }
  std::cout << "threadpool_size = " << threadpool_size << std::endl;

  GraphDef graph_def;
  Status status;

  // accept pb or pbtxt files
  if (model_path.find(".pbtxt") == std::string::npos) {
    status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
  } else {
    status = ReadTextProto(Env::Default(), model_path, &graph_def);
  }

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  TF_CHECK_OK(example::Test(graph_def, input_names, output_names, batch_size,
                            num_infers_per_thread, num_threads, use_xla,
                            threadpool_num, threadpool_size));

  return 0;
}
