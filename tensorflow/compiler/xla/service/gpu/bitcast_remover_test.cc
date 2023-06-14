/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/bitcast_remover.h"

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace {

class BitcastRemoverTest : public HloTestBase {
 public:
  void RunPassAndCheck(absl::string_view hlo_text, absl::string_view pattern) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_text));
    TF_ASSERT_OK(BitcastRemover().Run(module.get()).status());

    HloPrintOptions print_opts;
    print_opts.set_print_operand_shape(false);
    StatusOr<bool> filecheck_result =
        RunFileCheck(module->ToString(print_opts), pattern);
    TF_ASSERT_OK(filecheck_result.status());
    EXPECT_TRUE(filecheck_result.value());
  }
};

TEST_F(BitcastRemoverTest, OnlyReshape) {
  absl::string_view hlo_text = R"(
HloModule m

ENTRY e {
  a = f16[1,16,17,3]{3,2,1,0} parameter(0)
  ROOT b = f16[16,51]{1,0} bitcast(a)
}
)";

  RunPassAndCheck(hlo_text, R"(
; CHECK: %[[VAL_0:.*]] = f16[1,16,17,3]{3,2,1,0} parameter(0)
; CHECK: ROOT %{{.*}} = f16[16,51]{1,0} reshape(%[[VAL_0]])
)");
}

TEST_F(BitcastRemoverTest, OnlyReshape2) {
  absl::string_view hlo_text = R"(
HloModule m

ENTRY e {
  a = f16[17,3,1,16]{1,0,3,2} parameter(0)
  ROOT b = f16[51,16]{0,1} bitcast(a)
}
)";

  RunPassAndCheck(hlo_text, R"(
; CHECK: %[[VAL_0:.*]] = f16[17,3,1,16]{1,0,3,2} parameter(0)
; CHECK: ROOT %{{.*}} = f16[51,16]{0,1} reshape(%[[VAL_0]])
)");
}

TEST_F(BitcastRemoverTest, OnlyTranspose) {
  absl::string_view hlo_text = R"(
HloModule m

ENTRY e {
  a = s8[3,7,6,4]{3,2,1,0} parameter(0)
  ROOT b = s8[3,6,4,7]{2,1,3,0} bitcast(a)
}
)";

  RunPassAndCheck(hlo_text, R"(
; CHECK: %[[VAL_0:.*]] = s8[3,7,6,4]{3,2,1,0} parameter(0)
; CHECK: ROOT %{{.*}} = s8[3,6,4,7]{2,1,3,0} transpose(%[[VAL_0]]), dimensions={0,2,3,1}
)");
}

TEST_F(BitcastRemoverTest, ReshapeAndTranspose) {
  absl::string_view hlo_text = R"(
HloModule m

ENTRY e {
  a = s8[16,17,3]{2,1,0} parameter(0)
  ROOT b = s8[51,16]{0,1} bitcast(a)
}
)";

  RunPassAndCheck(hlo_text, R"(
; CHECK: %[[VAL_0:.*]] = s8[16,17,3]{2,1,0} parameter(0)
; CHECK: %[[VAL_1:.*]] = s8[16,51]{1,0} reshape(%[[VAL_0]])
; CHECK: ROOT %{{.*}} = s8[51,16]{0,1} transpose(%[[VAL_1]]), dimensions={1,0}
)");
}

TEST_F(BitcastRemoverTest, ReshapeAndTranspose2) {
  absl::string_view hlo_text = R"(
HloModule m

ENTRY e {
  a = s8[16,17,3,7]{3,2,1,0} parameter(0)
  ROOT b = s8[7,16,51]{0,2,1} bitcast(a)
}
)";

  RunPassAndCheck(hlo_text, R"(
; CHECK: %[[VAL_0:.*]] = s8[16,17,3,7]{3,2,1,0} parameter(0)
; CHECK: %[[VAL_1:.*]] = s8[16,51,7]{2,1,0} reshape(%[[VAL_0]])
; CHECK: ROOT %{{.*}} = s8[7,16,51]{0,2,1} transpose(%[[VAL_1]]), dimensions={2,0,1}
)");
}

TEST_F(BitcastRemoverTest, TransposeAndReshape) {
  absl::string_view hlo_text = R"(
HloModule m

ENTRY e {
  a = s8[16,3,17]{1,2,0} parameter(0)
  ROOT b = s8[51,16]{1,0} bitcast(a)
}
)";

  RunPassAndCheck(hlo_text, R"(
; CHECK: %[[VAL_0:.*]] = s8[16,3,17]{1,2,0} parameter(0)
; CHECK: %[[VAL_1:.*]]  = s8[16,17,3]{2,1,0} transpose(%[[VAL_0]]), dimensions={0,2,1}
; CHECK: ROOT %{{.*}} = s8[51,16]{1,0} reshape(%[[VAL_1]])
)");
}

TEST_F(BitcastRemoverTest, TransposeAndReshapeAndTranspose) {
  absl::string_view hlo_text = R"(
HloModule m

ENTRY e {
  a = s8[16,3,17]{1,2,0} parameter(0)
  ROOT b = s8[16,51]{0,1} bitcast(a)
}
)";

  RunPassAndCheck(hlo_text, R"(
; CHECK: %[[VAL_0:.*]] = s8[16,3,17]{1,2,0} parameter(0)
; CHECK: %[[VAL_1:.*]] = s8[16,17,3]{2,1,0} transpose(%[[VAL_0]]), dimensions={0,2,1}
; CHECK: %[[VAL_2:.*]] = s8[51,16]{1,0} reshape(%[[VAL_1]])
; CHECK: ROOT %{{.*}} = s8[16,51]{0,1} transpose(%[[VAL_2]]), dimensions={1,0}
)");
}

}  // namespace
}  // namespace xla
