/* Copyright 2025 The OpenXLA Authors.

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

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/tools/matmul_perf_table_gen.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

constexpr absl::string_view kUsageText = R"(
This tool runs specified matrix shapes and datatypes (HLO dots) on given hardware and
saves clock cycles for each. Matrix shapes can be specified by defining a search space.

Assume matrix multiplication dims: [n,k] @ [k,m] -> [n,m].

The specification has a format

{m,n,k}_spec='start=<start>,stop=<stop>,step=<step>'

Which means for a particular spec we will generate a set
  {<start> + n * <step> | n * <step> <= <stop> - <start> for every n w/ {0}}

Program expects a spec for every dim. The generated matrix multplication shapes
are a cartesian product of these three specs (+ specified data types).

Usage:

1. Run cartesian product for
  shape:{256x256x256} x dtype:{bf16,bf16->bf16 and bf16,bf16->f32} and dump to proto.

bazel run matmul_perf_table_gen_main --config=cuda -- \
  --alsologtostderr \
  --m_spec='start=256,stop=256,step=1' \
  --n_spec='start=256,stop=256,step=1' \
  --k_spec='start=256,stop=256,step=1' \
  --dtypes_spec='lhs=bf16,rhs=bf16,out=bf16;lhs=bf16,rhs=bf16,out=f32' \
  --output=/tmp/proto.pbtxt

cat /tmp/proto.pbtxt

entries {
  key: "sm_86"
  value {
    entries {
      instruction {
        name: "_"
        opcode: "dot"
        shape {
          element_type: BF16
          dimensions: 256
          dimensions: 256
        }
        ...
      }
      clock_cycles: 10022
    }
    entries {
      instruction {
        name: "_"
        opcode: "dot"
        shape {
          element_type: F32
          dimensions: 256
          dimensions: 256
       }
       ...
      clock_cycles: 10137
    }
  }
}

2. Run cartesian product for
  shape:{8x16x16 and 16x16x16 and 24x16x16} x dtype:{bf16,bf16->bf16} and print to stdout.
bazel run matmul_perf_table_gen_main --config=cuda -- \
  --alsologtostderr \
  --m_spec='start=8,stop=24,step=8' \
  --n_spec='start=16,stop=16,step=1' \
  --k_spec='start=16,stop=16,step=1' \
  --dtypes_spec='lhs=bf16,rhs=bf16,out=bf16 \

entries {
  key: "sm_90"
  value {
    entries {
      instruction {
        name: "_"
        opcode: "dot"
        shape {
          element_type: BF16
          dimensions: 24
          dimensions: 16
        }
      }
      ...
      clock_cycles: 10961
    }
    entries {
      instruction {
        name: "_"
        opcode: "dot"
        shape {
          element_type: BF16
          dimensions: 16
          dimensions: 16
        }
      }
      ...
      clock_cycles: 9440
    }
    entries {
      instruction {
        name: "_"
        opcode: "dot"
        shape {
          element_type: BF16
          dimensions: 8
          dimensions: 16
        }
      }
      ...
      clock_cycles: 9440
    }
  }
}
)";

using ::xla::gpu::MatmulPerfTableGen;

std::pair<std::string /*key*/, std::string /*value*/> ExtractKV(
    absl::string_view token_it, char elem_delim = '=') {
  std::string token = std::string(token_it);
  size_t delim_pos = token.find_first_of(elem_delim);
  CHECK_NE(delim_pos, std::string::npos);
  CHECK(delim_pos + 1 < token.size());
  std::string key = token.substr(0, delim_pos);
  std::string value = token.substr(delim_pos + 1);
  return {key, value};
}

MatmulPerfTableGen::StepSpec ParseSpec(absl::string_view spec,
                                       char elem_delim = ',') {
  MatmulPerfTableGen::StepSpec result;
  for (auto& token_it : absl::StrSplit(spec, elem_delim)) {
    auto [key, value] = ExtractKV(token_it);
    if (key == "start") {
      CHECK(absl::SimpleAtoi(value, &result.start));
    } else if (key == "stop") {
      CHECK(absl::SimpleAtoi(value, &result.stop));
    } else if (key == "step") {
      CHECK(absl::SimpleAtoi(value, &result.step));
    } else {
      LOG(FATAL) << "Cannot parse: " << spec;
    }
  }
  CHECK_LE(result.start, result.stop);
  CHECK_GT(result.step, 0);
  return result;
}

std::vector<MatmulPerfTableGen::DataTypeSpec> ParseDataTypes(
    absl::string_view types, char set_delim = ';', char elem_delim = ',') {
  std::vector<MatmulPerfTableGen::DataTypeSpec> result;
  for (auto& spec : absl::StrSplit(types, set_delim)) {
    MatmulPerfTableGen::DataTypeSpec spec_parsed;
    for (auto& token_it : absl::StrSplit(spec, elem_delim)) {
      auto [key, value] = ExtractKV(token_it);
      if (key == "lhs") {
        spec_parsed.lhs_dtype = value;
      } else if (key == "rhs") {
        spec_parsed.rhs_dtype = value;
      } else if (key == "out") {
        spec_parsed.out_dtype = value;
      } else {
        LOG(FATAL) << "Cannot parse: " << token_it;
      }
    }
    result.push_back(spec_parsed);
  }
  return result;
}

MatmulPerfTableGen::Config CreateConfig(absl::string_view m_spec,
                                        absl::string_view n_spec,
                                        absl::string_view k_spec,
                                        absl::string_view dtypes,
                                        absl::string_view output,
                                        bool dry_run) {
  MatmulPerfTableGen::Config cfg;

  // Search space.
  cfg.m_spec = ParseSpec(m_spec);
  cfg.n_spec = ParseSpec(n_spec);
  cfg.k_spec = ParseSpec(k_spec);
  cfg.dtypes = ParseDataTypes(dtypes);

  // Execution opts.
  cfg.dry_run = dry_run;
  cfg.output = output;
  return cfg;
}

// TODO(b/390097558): Sweep through minor and major dimensions for dots.
// TODO(b/390097558): Fetch dots from benchmarks.
int main(int argc, char* argv[]) {
  std::string m_spec;
  std::string n_spec;
  std::string k_spec;
  std::string dtypes;
  std::string out;
  bool dry_run = false;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("m_spec", &m_spec,
                "Spec for 'M' dimension. Format example: start=1,stop=4,step=2 "
                "generates {1,2,4}.'"),
      tsl::Flag("n_spec", &n_spec,
                "Spec for 'N' dimension. Format example: start=1,stop=4,step=2 "
                "generates {1,2,4}."),
      tsl::Flag("k_spec", &k_spec,
                "Spec for 'K' dimension. Format example: start=1,stop=4,step=2 "
                "generates {1,2,4}."),
      tsl::Flag("dtypes_spec", &dtypes,
                "Comma separated list of dtypes for which we will perform "
                "table gen."),
      tsl::Flag("output", &out,
                "Output mode. By default it's 'stdout'. If proto file is "
                "provided it's output will be merged (but not deduplicated) "
                "and rewritten with newly profiled ops. If proto file is "
                "provided but it does not exist a new one will be created."),
      tsl::Flag("dry_run", &dry_run,
                "For a defined search space does not perform measurements but "
                "runs everything else."),
  };
  const std::string kUsageString =
      absl::StrCat(kUsageText, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }

  MatmulPerfTableGen::Config cfg =
      CreateConfig(m_spec, n_spec, k_spec, dtypes, out, dry_run);
  MatmulPerfTableGen table_gen(std::move(cfg));
  xla::gpu::DeviceHloInstructionProfiles result = table_gen.ComputeTable();
  CHECK_OK(table_gen.Dump(result));

  return 0;
}
