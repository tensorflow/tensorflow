// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/xla/test.h"

using namespace poplar;
using namespace poplar::program;

namespace xla {
namespace poplarplugin {

const std::size_t S = 3;
const std::size_t P = 2;
const std::size_t Q = 2;

TEST(AddDynamicSliceTensorTest, Layout) {
  Device device = Device::createCPUDevice();
  Graph graph(device);

  Tensor t_layout;
  StatusOr<Tensor> t_status =
      AddDynamicSliceTensor(graph, "t", ShapeUtil::MakeShape(F32, {P, Q, S}),
                            ShapeUtil::MakeShape(F32, {P, Q, 1}), t_layout);

  ASSERT_TRUE(t_status.ok());

  Tensor t = t_status.ValueOrDie();

  graph.createHostWrite("t-write", t, true);
  graph.createHostRead("t_layout-read", t_layout, true);
  graph.createHostRead("t-read", t, true);

  auto prog = Sequence();

  Engine engine(graph, prog);
  engine.load(device);

  int i = 0;
  float write_buffer[P][Q][S];
  for (int s = 0; s < S; ++s) {
    for (int p = 0; p < P; ++p) {
      for (int q = 0; q < Q; ++q) {
        write_buffer[p][q][s] = static_cast<float>(i++);
      }
    }
  }

  engine.writeTensor("t-write", (void*)write_buffer);
  engine.run();

  float read_buffer[Q * P * S];
  engine.readTensor("t_layout-read", read_buffer);

  const float expected_buffer[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
  EXPECT_TRUE(
      std::equal(read_buffer, read_buffer + (P * Q * S), expected_buffer));
}

}  // namespace poplarplugin
}  // namespace xla
