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

// The builtin inputs provide a mechanism to generate simple TensorFlow graphs
// and feed them as inputs to Grappler. This can be used for quick experiments
// or to derive small regression tests.

#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"

namespace tensorflow {
namespace grappler {

// Make a program with specified number of stages and "width" ops per stage.
namespace {
GraphDef CreateGraphDef(int num_stages, int width, int tensor_size,
                        bool use_multiple_devices, bool insert_queue,
                        const std::vector<string>& device_names) {
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  // x is from the feed.
  const int batch_size = tensor_size < 0 ? 1 : tensor_size;
  Output x = RandomNormal(s.WithOpName("x").WithDevice("/CPU:0"),
                          {batch_size, 1}, DataType::DT_FLOAT);

  // Create stages.
  std::vector<Output> last_stage;
  last_stage.push_back(x);
  for (int i = 0; i < num_stages; i++) {
    std::vector<Output> this_stage;
    for (int j = 0; j < width; j++) {
      if (last_stage.size() == 1) {
        Output unary_op =
            Sign(s.WithDevice(
                     device_names[use_multiple_devices ? j % device_names.size()
                                                       : 0]),
                 last_stage[0]);
        this_stage.push_back(unary_op);
      } else {
        Output combine =
            AddN(s.WithDevice(
                     device_names[use_multiple_devices ? j % device_names.size()
                                                       : 0]),
                 last_stage);
        this_stage.push_back(combine);
      }
    }
    last_stage = this_stage;
  }

  if (insert_queue) {
    FIFOQueue queue(s.WithOpName("queue").WithDevice("/CPU:0"),
                    {DataType::DT_FLOAT});
    QueueEnqueue enqueue(s.WithOpName("enqueue").WithDevice("/CPU:0"), queue,
                         last_stage);
    QueueDequeue dequeue(s.WithOpName("dequeue").WithDevice("/CPU:0"), queue,
                         {DataType::DT_FLOAT});
    QueueClose cancel(s.WithOpName("cancel").WithDevice("/CPU:0"), queue,
                      QueueClose::CancelPendingEnqueues(true));
    last_stage = {dequeue[0]};
  }

  // Create output.
  AddN output(s.WithOpName("y").WithDevice("/CPU:0"), last_stage);

  GraphDef def;
  TF_CHECK_OK(s.ToGraphDef(&def));
  return def;
}
}  // namespace

TrivialTestGraphInputYielder::TrivialTestGraphInputYielder(
    int num_stages, int width, int tensor_size, bool insert_queue,
    const std::vector<string>& device_names)
    : num_stages_(num_stages),
      width_(width),
      tensor_size_(tensor_size),
      insert_queue_(insert_queue),
      device_names_(device_names) {}

bool TrivialTestGraphInputYielder::NextItem(GrapplerItem* item) {
  GrapplerItem r;
  r.id = strings::StrCat("ns:", num_stages_, "/",  // wrap
                         "w:", width_, "/",        // wrap
                         "ts:", tensor_size_);
  r.graph = CreateGraphDef(num_stages_, width_, tensor_size_,
                           true /*use_multiple_devices*/, insert_queue_,
                           device_names_);
  // If the batch size is variable, we need to choose a value to create a feed
  const int batch_size = tensor_size_ < 0 ? 1 : tensor_size_;
  Tensor x(DT_FLOAT, TensorShape({batch_size, 1}));
  r.feed.push_back(std::make_pair("x", x));
  r.fetch.push_back("y");

  if (insert_queue_) {
    QueueRunnerDef queue_runner;
    queue_runner.set_queue_name("queue");
    queue_runner.set_cancel_op_name("cancel");
    *queue_runner.add_enqueue_op_name() = "enqueue";
    r.queue_runners.push_back(queue_runner);
  }

  *item = std::move(r);
  return true;
}

}  // end namespace grappler
}  // end namespace tensorflow
