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

#include "absl/base/no_destructor.h"
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
    {{ 1018.0043402777778, 1023.5008680555555, 1025.861328125, 6.616354874682953, 6.620083771919308, 6.63456382646144 }},
    /*scale=*/
    {{ 590.632963260763, 593.7150826714844, 588.1463320809286, 0.9965492348670616, 1.0053835765793744, 0.9754062267657724 }}
  },
  /*hidden_layer_1=*/{
    /*weights=*/{{
      {{ 1.005891244892035, -0.11072959330426384, -0.968046373485514, -4.921521725424851, -0.13899276332759042, -1.226293159684938 }},
      {{ -1.2502714170922153, -1.1765486776310423, 0.4888716998554182, 0.5172335453177302, -0.593110373636639, 0.1209811095977735 }},
      {{ -0.18095204827877465, -0.06060134530184746, -0.3750576441489519, 0.8188111073502161, 0.3576805251431741, 1.4968701417615413 }},
      {{ -0.05111798751008719, 0.07769076545705815, 0.6207635489778949, 3.6101378942470967, -0.05282332398428117, 1.7854763674893495 }},
      {{ 0.747350403236783, 0.04930450878767915, 0.8777099456443157, 1.4990255665580505, 0.08212232867228593, -1.9843375290758656 }},
      {{ 0.5541966277545818, -0.33159380874523126, -0.3194666063636952, 1.6816423644238223, 0.1017686813844035, 3.1265875450394813 }},
      {{ 0.7345608720339312, 1.5974970323865725, -2.2148504671311953, 0.5363830361034387, 0.3700332861259481, 0.9331234440564571 }},
      {{ 1.4423494345704555, 0.289382200477869, 1.5110082992053955, 1.418682530523278, -0.5749656156984766, 2.684031259217246 }},
    }},
    /*biases=*/{
      { 1.0309774603375907, 0.6465236073322296, 3.553393762532912, -0.8185749009865861, -0.12955426436728715, 0.8031554507587597, -1.5785991424170187, -0.39445212677049063 }
    }
  },
  /*hidden_layer_2=*/{
    /*weights=*/{{
      {{ 0.5283786603352238, -0.056889406800860146, 0.6606078794503449, 0.48144968239995534, 0.010554846273878095, -0.14799959162846965, 0.20457525406298369, -0.12661568456264205 }},
      {{ -0.9534585055098438, -0.10012840023501332, -0.5795955688342004, 1.9348789870642515, 0.21271731229957186, 0.1557077642737526, 1.5288709351139655, -0.7002696129400411 }},
      {{ 2.1906088364770784, 0.07703067791952263, 2.1646753484073655, -0.32251840895547773, 0.6850197180505169, 0.30061421444806946, 1.0728025841881765, -0.8100244450669523 }},
      {{ 0.8106365294738354, -0.3410735969241413, 0.7910924608271775, -0.07017938451436888, 0.16051916138347214, 0.3004275708609215, 0.7729045870717262, -0.2332237341201925 }},
      {{ 0.07246756191918226, -0.05758991153686244, 0.911745169839482, 1.2510377533921035, -1.182537901423303, -1.295182969102456, -1.904956642808503, 0.007010431897136803 }},
      {{ -0.8733376736355871, 0.157671979745821, -1.0372041545921873, 3.3000069112365584, 0.25551941086911717, 0.9589328110123956, -0.23856740081287128, -2.0315351809352586 }},
      {{ -1.9337567589656532, 1.4676259894002257, 1.4886721579905993, -1.5705845737356183, 0.48937401463732866, 1.083620050144208, -1.0031665521883135, -1.2660789079048749 }},
      {{ 2.4310441535271687, 0.49189784223311395, 0.31483156395428413, -0.0865355927145238, 0.7631527157107736, -2.7077958375575055, -0.9228446079924654, -0.7391110100336947 }}
    }},
    /*biases=*/{
      { 0.3486705866196229, 2.2776343748673153, 0.10764831721796844, 0.09166185120840216, -1.0034214854612917, -0.927160996221299, 1.5172112381212808, 0.4772212967805247 }
    }
  },
  /*output_layer=*/{
    /*weights=*/{{
      {{ -0.5896335640757622, 2.3283855361577914, -1.9754605319484158, -0.6632271049751296, 1.9086390756642784, 3.7433099466616238, 1.824432545010804, 2.09625742741301 }}
    }},
    /*biases=*/{
      { -0.04977871506692414 }
    }
  },
  /*threshold=*/0.03,
};

static constexpr GemmFilter AMDGenoaGemmFilter{
  /*input_range=*/{16, 2048},
  /*lhs_dtype=*/PrimitiveType::F32,
  /*rhs_dtype=*/PrimitiveType::F32,
  /*out_dtype=*/PrimitiveType::F32,
};

static constexpr Net AMDGenoaNet {
  /*scaler=*/{
    /*mean=*/
    {{ 1002.1257625527921, 1040.082590333177, 1029.2266541529798, 6.598251685888309, 6.6404292804559635, 6.654739928938151 }},
    /*scale=*/
    {{ 594.7876923848536, 596.4073362076931, 579.6412706089599, 0.9806421276088567, 1.0047695259963145, 0.9426741875163773 }}
  },
  /*hidden_layer_1=*/{
    /*weights=*/{{
      {{ -1.8525957121690557, -0.12363449157789586, -0.8386369273170659, -3.3954448946951414, 0.33886983245776847, -0.9317201523227778 }},
      {{ 0.5120871227810689, -1.5214338320394882, -0.7274651337778577, 1.3200335397974383, 0.5923648903998096, 1.7241300558638806 }},
      {{ -1.8962086199924455, -1.7249686491398133, -0.15047446639707035, 1.1356266853003538, 2.70970817913586, 0.7154911861570797 }},
      {{ -0.0020642101422613684, 0.14097136562712495, -0.09163191463561046, -0.02820803725731568, 0.03955304538877561, 0.16661173026752624 }},
      {{ -2.001943294276495, 1.4970193262821043, 0.7777143265827485, 2.4639584696544814, -2.5991183905189827, -1.7850169579413313 }},
      {{ 1.447549192320386, 1.361173224199936, -0.43481695242532376, 0.8671054211799711, -0.25280176689036743, 2.5275631466098756 }},
      {{ -0.1441737229514477, -0.3347815066193075, -0.4611605626954958, 2.181799166007114, 1.6323192872442907, 0.15928222502988382 }},
      {{ -2.4693645353916667, 2.0428513209522623, 1.749924045625967, 1.7019213519700969, 1.142024009890298, 0.8079292507334053 }},
    }},
    /*biases=*/{
      { 1.5113512311183828, 1.1809617131571573, 0.4478042727248335, -0.7947383308647096, 0.9825820925820611, 2.3808221280278645, -0.4758496641489877, -0.17036794796484162 }
    }
  },
  /*hidden_layer_2=*/{
    /*weights=*/{{
      {{ 0.35014293742424724, -0.5476873131211425, 2.120121365055135, 0.2526392749218728, -0.9407987661758053, -0.9846910595845514, 0.48052568401528156, 0.3180570478836074 }},
      {{ -0.7150926363558526, -0.8982987199655207, 1.5360045616379934, 0.1803870098694977, -3.472436216409119, 1.0044744229912244, 0.4335209413598261, 1.041064125260285 }},
      {{ 0.8966091979738551, 2.227321493016081, -1.4494415364024744, 0.0254402271585605, 1.537814361721512, 1.3889169541326774, -0.16388027659485285, 1.7176584253327756 }},
      {{ -0.0597193070244571, -2.5061384854500623, 2.124958864280429, 0.020011322193699232, 0.48193219767756923, 1.323504600240459, 1.9316400738347115, 0.9369893381605147 }},
      {{ 0.032699385072007786, -0.9107111450265581, 0.5333081096223299, -0.03148698112648736, -0.16377114190322742, -1.6085833510134837, 0.7968443974614786, 2.238778726588351 }},
      {{ -1.5463140026380815, 1.5303232342633923, 1.3050928202993113, 0.17310173303944465, -0.9857258240351916, -0.9550401224056498, -0.0071639117187034555, -0.7976265480773017 }},
      {{ -2.2035284133385966, -0.9376946654220445, 1.3260159557423135, -0.406540234329094, 0.9533782897195586, 2.4910588396009277, -1.3676852361286786, 0.6322307655713885 }},
      {{ 0.32147827378305643, 0.26619936190034904, -1.200452781975383, 0.00920972156405369, 2.8156283745109, -1.229898728424551, 1.3389598114381762, 1.7166384524807266 }},
    }},
    /*biases=*/{
      { 0.9699621387917686, 0.5909615635903002, 1.427690501080743, 0.8892984037070786, -0.8768819880157831, 3.2935191809063777, 1.8570755726618182, -1.0853011858156631 }
    }
  },
  /*output_layer=*/{
    /*weights=*/{{
      {{ 2.253756860038231, 2.6258281376128605, -1.6079113669729557, -4.006565437256236, -2.8600538503590087, 2.808280959332861, 2.8146068804622777, 2.604358926679522 }},
    }},
    /*biases=*/{
      { 0.26142347527936727 }
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

bool IsAMDGenoa(const llvm::TargetMachine* target_machine) {
  CHECK(target_machine);
  return target_machine->getTargetCPU() == "znver4";
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

  if (IsAMDGenoa(cpu_features->target_machine()) && AMDGenoaGemmFilter(gemm)) {
    int out = AMDGenoaNet(gemm.dot_canonical_dims.m, gemm.dot_canonical_dims.k,
                          gemm.dot_canonical_dims.n);
    return out == 1 ? XnnGemmConfig::Opinion::kAccept
                    : XnnGemmConfig::Opinion::kReject;
  }

  return XnnGemmConfig::Opinion::kNoIdea;
}

const XnnGemmConfig& GetXnnGemmConfig() {
  static const absl::NoDestructor<XnnGemmConfig> gemm_config;
  return *gemm_config;
}

}  // namespace xla::cpu
