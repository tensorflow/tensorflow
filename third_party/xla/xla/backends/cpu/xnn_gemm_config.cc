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

#include "xla/backends/cpu/xnn_gemm_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

#include "absl/log/check.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

namespace {

double Relu(double x) { return std::max(0.0, x); }

template <size_t Size>
std::array<double, Size> Relu(const std::array<double, Size>& input) {
  std::array<double, Size> output{};
  for (size_t i = 0; i < Size; ++i) {
    output[i] = Relu(input[i]);
  }
  return output;
}

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double Sigmoid(std::array<double, 1> input) { return Sigmoid(input[0]); }

template <size_t InSize, size_t OutSize>
struct Layer {
  std::array<std::array<double, InSize>, OutSize> weights;
  std::array<double, OutSize> biases;

  std::array<double, OutSize> operator()(
      const std::array<double, InSize>& input) const {
    std::array<double, OutSize> output{};
    for (size_t i = 0; i < OutSize; ++i) {
      output[i] = std::inner_product(input.begin(), input.end(),
                                     weights[i].begin(), 0.0);
      output[i] += biases[i];
    }
    return output;
  }
};

template <size_t InSize>
struct Scaler {
  std::array<double, InSize> mean;
  std::array<double, InSize> scale;

  std::array<double, InSize> operator()(
      const std::array<double, InSize>& features) const {
    std::array<double, InSize> out;
    for (size_t i = 0; i < features.size(); ++i) {
      out[i] = (features[i] - mean[i]) / scale[i];
    }
    return out;
  }
};

std::array<double, 6> ExtractFeatures(int m, int k, int n) {
  std::array<double, 6> features = {static_cast<double>(m),
                                    static_cast<double>(k),
                                    static_cast<double>(n),
                                    std::log(m),
                                    std::log(k),
                                    std::log(n)};
  return features;
}

struct Net {
  static constexpr size_t kNumFeatures = 6;
  static constexpr size_t kHiddenLayer1Size = 8;
  static constexpr size_t kHiddenLayer2Size = 8;

  Scaler<kNumFeatures> scaler;
  Layer<kNumFeatures, kHiddenLayer1Size> hidden_layer_1;
  Layer<kHiddenLayer1Size, kHiddenLayer2Size> hidden_layer_2;
  Layer<kHiddenLayer2Size, 1> output_layer;
  double threshold;

  int operator()(double m, double k, double n) const {
    std::array<double, kNumFeatures> features = ExtractFeatures(m, k, n);
    double probability = Sigmoid(output_layer(
        Relu(hidden_layer_2(Relu(hidden_layer_1(scaler(features)))))));
    return probability < threshold ? 1 : 0;
  }
};

struct Range {
  int min;
  int max;

  template <class... Args>
  bool Contains(Args... args) const {
    auto check = [this](int x) -> bool { return min <= x && x <= max; };
    return (check(args) && ...);
  }
};

struct GemmFilter {
  Range input_range;
  PrimitiveType lhs_dtype;
  PrimitiveType rhs_dtype;
  PrimitiveType out_dtype;

  bool operator()(const XnnGemm& gemm) const {
    return input_range.Contains(gemm.dot_canonical_dims.m,
                                gemm.dot_canonical_dims.k,
                                gemm.dot_canonical_dims.n) &&
           gemm.lhs_dtype == lhs_dtype && gemm.rhs_dtype == rhs_dtype &&
           gemm.out_dtype == out_dtype &&
           gemm.dot_canonical_dims.lhs_canonical &&
           !gemm.dot_canonical_dims.rhs_column_major &&
           gemm.dot_canonical_dims.rhs_canonical &&
           !gemm.dot_canonical_dims.output_column_major;
  }
};

// NOLINTBEGIN
// clang-format off

static constexpr GemmFilter BF16BF16F32GemmFilter{
  /*input_range=*/{0, std::numeric_limits<int>::max()},
  /*lhs_dtype=*/PrimitiveType::BF16,
  /*rhs_dtype=*/PrimitiveType::BF16,
  /*out_dtype=*/PrimitiveType::F32,
};

static constexpr GemmFilter AMDRomeGemmFilter{
  /*input_range=*/{16, 2048},
  /*lhs_dtype=*/PrimitiveType::F32,
  /*rhs_dtype=*/PrimitiveType::F32,
  /*out_dtype=*/PrimitiveType::F32,
};

static constexpr Net AMDRomeNet{
  /*scaler=*/{
    /*mean=*/
    {{ 1260.176782688408, 1022.4904580682796, 1020.0638351065307, 6.866230392543272, 6.6219307914483245, 6.625439794455572 }},
    /*scale=*/
    {{ 596.0724542398273, 592.3072052082454, 590.0107526796818, 0.9781757980565287, 0.9934520144953549, 0.9767752827270596 }}
  },
  /*hidden_layer_1=*/{
    /*weights=*/{{
      {{ -1.1067415056207293, -1.5768032556970715, -1.857432446815887, -2.274357938055606, 0.20072093031321048, 0.42525858635075686 }},
      {{ 0.6641830717648227, -0.1348693740866461, 0.6503311504843254, -1.0644659548831132, 0.2773945732210612, -2.2937718186533798 }},
      {{ -1.9952103909464958, -0.3598121541710137, -1.517300213597801, 0.1500046740980615, 0.03843929168081172, 2.4381868146886236 }},
      {{ -0.552506077401237, 0.19843349614999084, -0.9693656867138434, 2.836596915608218, -0.056741365402952736, 2.74752722582657 }},
      {{ -0.8307866265141837, -0.27059731547224314, 1.078076654323506, 2.024224405334713, -2.3276163078716987, 1.6221199123650774 }},
      {{ 1.149439782090887, 0.17964130912962506, -0.226600733139848, -1.0659592041949355, -0.08637229703406037, -2.4386823311144736 }},
      {{ -0.2038246372813593, -0.3200350290868884, 1.3294209853943872, -0.09661218361772743, 0.2568191141244127, 0.3694445094812476 }},
      {{ 0.22468914121163505, 0.4068312248990986, -0.049538864830452244, 1.708300900954308, 1.4861437729507463, 2.9187563082847903 }},
    }},
    /*biases=*/{
      { 1.3693599531845795, -2.0426696231011854, 2.7265313071818067, -0.19845870611878083, 0.55496647188004, 0.14663669973642213, -1.233470943433025, 0.1598907831069279 }
    }
  },
  /*hidden_layer_2=*/{
    /*weights=*/{{
      {{ -1.33906713654116, -2.482296547296389, 1.4552712398855803, 1.8497794226209343, -0.7588134640967776, 0.7748180344911018, 0.49831179993441294, 0.6566250444087962 }},
      {{ -1.0574231065146888, 0.3409066367887875, 0.4578212517905438, 0.7788360404620238, -0.7741476090932119, -0.5976010810458275, 0.7987696949835748, -1.0459260220130473 }},
      {{ 0.17004942931368353, -0.4728810321390739, 0.06142214094749654, 0.6660267904693781, -0.6601384407951393, -1.0349097261944356, 1.1436553921145678, 1.6165317738668088 }},
      {{ -0.979848643839785, 0.7813133650835401, 1.7891619147038698, 0.18627412613950164, -0.13274493579979216, 0.8419967198728521, -0.2172673853587043, 0.18441176576034882 }},
      {{ -0.505714104889808, 0.34419793988635666, 0.4152800608111852, -2.252916551136698, -0.06927487291604531, -0.5572530294844231, -0.04169721307653461, 0.7807841718137976 }},
      {{ -1.2814324297453774, 0.7112981365639622, 1.6523811849381065, -1.1985169999443233, 0.07308553077894221, 0.9747514576151951, -0.3672023577904536, -0.7653262828445461 }},
      {{ -1.2756750826090897, 1.8883867653074107, 1.3941120682125925, -1.8465512314445875, 0.0930160928685092, 0.07506592537860401, -0.819220668755894, 2.4866833693870953 }},
      {{ 1.4722950809666655, -1.4181037012833242, -1.8434961841114577, -0.9187090262731159, 0.32224375954731765, -1.0144295875820903, 0.3035921228235539, -0.10734242103774676 }},
    }},
    /*biases=*/{
      { -0.697454441700992, 3.473046282613419, 2.4595894594124, -1.5006708103518294, 1.0823130570296722, -2.0834873171432395, 1.6737091528663508, -2.172967611312064 }
    }
  },
  /*output_layer=*/{
    /*weights=*/{{
      {{ -2.8824937559983437, 3.145011777850824, -1.4771950602925645, -1.890715798311394, -1.2909523481754717, -1.539182156545836, 3.293356714639934, 2.5281957171296656 }}
    }},
    /*biases=*/{
      { 0.9506203943504298 }
    }
  },
  /*threshold=*/0.03,
};
// clang-format on
// NOLINTEND

bool IsAMDRome(const llvm::TargetMachine* target_machine) {
  CHECK(target_machine);
  return target_machine->getTargetCPU() == "znver2";
}

bool IsAMDMilan(const llvm::TargetMachine* target_machine) {
  CHECK(target_machine);
  return target_machine->getTargetCPU() == "znver3";
}

}  // namespace

XnnGemmConfig::Opinion XnnGemmConfig::Evaluate(
    const XnnGemm& gemm, const TargetMachineFeatures* cpu_features) const {
  if (test_filter_) {
    return test_filter_(gemm) ? XnnGemmConfig::Opinion::kAccept
                              : XnnGemmConfig::Opinion::kReject;
  }

  if (!cpu_features || !cpu_features->target_machine()) {
    return XnnGemmConfig::Opinion::kNoIdea;
  }

  CHECK(cpu_features);
  CHECK(cpu_features->target_machine());

  if (BF16BF16F32GemmFilter(gemm)) {
    return XnnGemmConfig::Opinion::kAccept;
  }

  if ((IsAMDRome(cpu_features->target_machine()) ||
       IsAMDMilan(cpu_features->target_machine())) &&
      AMDRomeGemmFilter(gemm)) {
    int out = AMDRomeNet(gemm.dot_canonical_dims.m, gemm.dot_canonical_dims.k,
                         gemm.dot_canonical_dims.n);
    return out == 1 ? XnnGemmConfig::Opinion::kAccept
                    : XnnGemmConfig::Opinion::kReject;
  }

  return XnnGemmConfig::Opinion::kNoIdea;
}

const XnnGemmConfig& GetXnnGemmConfig() {
  static const XnnGemmConfig* gemm_config = new XnnGemmConfig();
  return *gemm_config;
}

}  // namespace xla::cpu
