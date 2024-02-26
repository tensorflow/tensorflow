/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/mock_nccl_xml.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/status.h"
#include "tsl/platform/test.h"
#if GOOGLE_CUDA
#include "third_party/gpus/nccl/graph/xml.h"
#endif

namespace xla {
namespace gpu {
namespace {

#if GOOGLE_CUDA

class MockNcclXmlParserTest : public ::testing::Test {};

TEST_F(MockNcclXmlParserTest, PciNic) {
  const std::string original = R"(
    <pci busid="0000:0c:00.0" class="0x020700" vendor="0x15b3" device="0x101b" subsystem_vendor="0x15b3" subsystem_device="0x0007" link_speed="16.0 GT/s PCIe" link_width="16">
      <nic>
        <net name="mlx5_0" dev="0" speed="200000" port="1" latency="0.000000" guid="0xcf70b0003a1420c" maxconn="131072" gdr="1"/>
      </nic>
    </pci>
  )";
  auto xml = std::make_unique<ncclXml>();
  auto result = MockTopoGetXml(original, xml.get());

  EXPECT_EQ(OkStatus(), result);
  EXPECT_EQ(xml->maxIndex, 3);
  EXPECT_EQ(std::string(xml->nodes[0].name), "pci");
  EXPECT_EQ(xml->nodes[0].nAttrs, 8);
  EXPECT_EQ(std::string(xml->nodes[0].attrs[0].key), "busid");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[0].value), "0000:0c:00.0");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[1].key), "class");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[1].value), "0x020700");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[2].key), "vendor");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[2].value), "0x15b3");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[3].key), "device");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[3].value), "0x101b");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[4].key), "subsystem_vendor");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[4].value), "0x15b3");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[5].key), "subsystem_device");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[5].value), "0x0007");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[6].key), "link_speed");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[6].value), "16.0 GT/s PCIe");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[7].key), "link_width");
  EXPECT_EQ(std::string(xml->nodes[0].attrs[7].value), "16");
  EXPECT_EQ(xml->nodes[0].nSubs, 1);
  EXPECT_EQ(std::string(xml->nodes[0].subs[0]->name), "nic");
}

TEST_F(MockNcclXmlParserTest, GpuNvlink) {
  const std::string original = R"(
    <gpu dev="0" sm="80" rank="0" gdr="1">
      <nvlink target="0000:c7:00.0" count="2" tclass="0x068000"/>
    </gpu>
  )";
  auto xml = std::make_unique<ncclXml>();
  auto result = MockTopoGetXml(original, xml.get());
  EXPECT_EQ(OkStatus(), result);
  EXPECT_EQ(xml->maxIndex, 2);
  EXPECT_EQ(std::string(xml->nodes[0].name), "gpu");
  EXPECT_EQ(xml->nodes[0].nAttrs, 4);
  EXPECT_EQ(xml->nodes[0].nSubs, 1);
  EXPECT_EQ(std::string(xml->nodes[0].subs[0]->name), "nvlink");
  EXPECT_EQ(xml->nodes[0].subs[0]->nAttrs, 3);
  EXPECT_EQ(std::string(xml->nodes[0].subs[0]->attrs[0].key), "target");
  EXPECT_EQ(std::string(xml->nodes[0].subs[0]->attrs[0].value), "0000:c7:00.0");
  EXPECT_EQ(std::string(xml->nodes[0].subs[0]->attrs[1].key), "count");
  EXPECT_EQ(std::string(xml->nodes[0].subs[0]->attrs[1].value), "2");
  EXPECT_EQ(std::string(xml->nodes[0].subs[0]->attrs[2].key), "tclass");
  EXPECT_EQ(std::string(xml->nodes[0].subs[0]->attrs[2].value), "0x068000");
}

#endif

}  // namespace
}  // namespace gpu
}  // namespace xla
