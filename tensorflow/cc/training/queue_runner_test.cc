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

#include "tensorflow/cc/training/queue_runner.h"
#include <string>
#include <vector>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/queue_runner.pb.h"
#include "tensorflow/core/public/session.h"

namespace {

using ::tensorflow::DataType;
using ::tensorflow::error::Code;
using ::tensorflow::GraphDef;
using ::tensorflow::ops::Assign;
using ::tensorflow::ops::Const;
using ::tensorflow::ops::CountUpTo;
using ::tensorflow::ops::FIFOQueue;
using ::tensorflow::ops::InputList;
using ::tensorflow::ops::QueueClose;
using ::tensorflow::ops::QueueDequeue;
using ::tensorflow::ops::QueueEnqueue;
using ::tensorflow::ops::Square;
using ::tensorflow::ops::Variable;
using ::tensorflow::QueueRunner;
using ::tensorflow::QueueRunnerDef;
using ::tensorflow::Scope;
using ::tensorflow::Session;
using ::tensorflow::SessionOptions;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;

constexpr char kAssignOpName[] = "assign";
constexpr char kCountUpToOpName[] = "count";
constexpr char kIllegalOpName1[] = "would fail";
constexpr char kIllegalOpName2[] = "fail again";
constexpr char kQueueName[] = "unit_test";
constexpr char kSquareOpName[] = "square";
constexpr char kVarOpName[] = "var";

GraphDef BuildSimpleGraph() {
  Scope root = Scope::NewRootScope();
  auto init_value = Const(root, 0);
  auto var = Variable(root.WithOpName(kVarOpName), TensorShape({}),
                      DataType::DT_INT32);
  auto assign = Assign(root.WithOpName(kAssignOpName), var, init_value);
  auto count = CountUpTo(root.WithOpName(kCountUpToOpName), var, 10);
  Square(root.WithOpName(kSquareOpName), var);  // NOLINT

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));
  return graph_def;
}

QueueRunnerDef BuildQueueRunnerDef(
    const std::string& queue_name, const std::vector<std::string>& enqueue_ops,
    const std::string& close_op,
    const std::vector<Code>& queue_closed_error_codes) {
  QueueRunnerDef queue_runner_def;
  *queue_runner_def.mutable_queue_name() = kQueueName;
  for (const std::string& enqueue_op : enqueue_ops) {
    *queue_runner_def.mutable_enqueue_op_name()->Add() = enqueue_op;
  }
  *queue_runner_def.mutable_close_op_name() = close_op;
  for (const auto& error_code : queue_closed_error_codes) {
    *queue_runner_def.mutable_queue_closed_exception_types()->Add() =
        error_code;
  }
  return queue_runner_def;
}

std::unique_ptr<Session> BuildSessionAndInitVariable(
    const GraphDef& graph_def) {
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));

  std::vector<Tensor> nothing;
  TF_CHECK_OK(session->Run({}, {}, {kAssignOpName}, &nothing));
  return session;
}

TEST(QueueRunnerTest, BasicTest) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);

  QueueRunnerDef queue_runner_def = BuildQueueRunnerDef(
      kQueueName, {kCountUpToOpName, kCountUpToOpName}, kSquareOpName, {});

  QueueRunner qr(queue_runner_def);
  qr.Start(session.get());
  TF_EXPECT_OK(qr.Join());

  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session->Run({}, {kSquareOpName}, {}, &outputs));
  int square_value = *outputs[0].scalar<int>().data();
  EXPECT_EQ(square_value, 100);
}

TEST(QueueRunnerTest, QueueClosedCode) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);

  QueueRunnerDef queue_runner_def =
      BuildQueueRunnerDef(kQueueName, {kCountUpToOpName}, kSquareOpName,
                          {Code::OUT_OF_RANGE, Code::CANCELLED});

  QueueRunner qr(queue_runner_def);
  qr.Start(session.get());
  TF_EXPECT_OK(qr.Join());

  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session->Run({}, {kSquareOpName}, {}, &outputs));
  int square_value = *outputs[0].scalar<int>().data();
  EXPECT_EQ(square_value, 100);
}

TEST(QueueRunnerDef, CatchErrorInJoin) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);

  QueueRunnerDef queue_runner_def = BuildQueueRunnerDef(
      kQueueName, {kIllegalOpName1, kIllegalOpName2}, kCountUpToOpName, {});

  QueueRunner qr(queue_runner_def);
  qr.Start(session.get());
  EXPECT_EQ(qr.Join().code(), Code::NOT_FOUND);
}

TEST(QueueRunnerTest, RealEnqueueDequeue) {
  Scope root = Scope::NewRootScope();
  auto q0 = FIFOQueue(root.WithOpName("q0"), {DataType::DT_INT32});
  auto ten = Const(root, 10);
  auto enqueue0 = QueueEnqueue(root.WithOpName("enqueue0"), q0, {ten});
  auto close0 = QueueClose(root.WithOpName("close0"), q0);
  auto q1 = FIFOQueue(root.WithOpName("q1"), {DataType::DT_INT32});
  auto dequeue0 =
      QueueDequeue(root.WithOpName("dequeue0"), q0, {DataType::DT_INT32});
  auto enqueue1 = QueueEnqueue(root.WithOpName("enqueue1"), q1, {dequeue0[0]});
  auto dequeue1 =
      QueueDequeue(root.WithOpName("dequeue1"), q1, {DataType::DT_INT32});
  auto close1 = QueueClose(root.WithOpName("close1"), q1);

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));

  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));

  QueueRunnerDef queue_runner_def =
      BuildQueueRunnerDef(kQueueName, {"enqueue1"}, "close1", {});
  QueueRunner qr;
  qr.Init(queue_runner_def);
  TF_CHECK_OK(qr.Start(session.get()));

  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session->Run({}, {}, {"enqueue0"}, &outputs));
  TF_EXPECT_OK(session->Run({}, {}, {"enqueue0"}, &outputs));
  TF_EXPECT_OK(session->Run({}, {}, {"close0"}, &outputs));

  TF_EXPECT_OK(qr.Join());
  std::vector<Tensor> dq1;
  TF_EXPECT_OK(session->Run({}, {"dequeue1"}, {}, &dq1));
  EXPECT_EQ(*dq1[0].scalar<int>().data(), 10);
  std::vector<Tensor> dq2;
  TF_EXPECT_OK(session->Run({}, {"dequeue1"}, {}, &dq2));
  EXPECT_EQ(*dq2[0].scalar<int>().data(), 10);

  EXPECT_EQ(session->Run({}, {"dequeue1"}, {}, &dq1).code(),
            Code::OUT_OF_RANGE);
}

TEST(QueueRunnerTest, EmptyEnqueueOps) {
  QueueRunnerDef queue_runner_def =
      BuildQueueRunnerDef(kQueueName, {}, kCountUpToOpName, {});

  QueueRunner qr;
  EXPECT_EQ(qr.Init(queue_runner_def).code(), Code::INVALID_ARGUMENT);
}

TEST(QueueRunnerTest, InitAfterStart) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);
  QueueRunnerDef queue_runner_def =
      BuildQueueRunnerDef(kQueueName, {kCountUpToOpName}, kCountUpToOpName, {});

  QueueRunner qr;
  TF_EXPECT_OK(qr.Init(queue_runner_def));
  TF_EXPECT_OK(qr.Start(session.get()));
  EXPECT_EQ(qr.Init(queue_runner_def).code(), Code::ALREADY_EXISTS);
}

}  // namespace
