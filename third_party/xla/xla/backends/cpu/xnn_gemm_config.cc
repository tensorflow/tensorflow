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
  /*input_range=*/{16, 4096},
  /*lhs_dtype=*/PrimitiveType::F32,
  /*rhs_dtype=*/PrimitiveType::F32,
  /*out_dtype=*/PrimitiveType::F32,
};

static constexpr Net AMDRomeNet{
  /*scaler=*/{
    /*mean=*/
    {{ 2031.4479060265578, 2036.3171603677222, 2062.2170582226763, 7.29227087924762, 7.308301476602625, 7.331674465299577 }},
    /*scale=*/
    {{ 1188.2177375470617, 1178.7350461452038, 1179.7790996965598, 1.0416890873676914, 1.0053399234375506, 0.9757991392501179 }},
  },
  /*hidden_layer_1=*/{
    /*weights=*/{{
      {{ 0.5255922128957278, -0.8065013670906714, -0.5264014380189966, -1.2772498330118651, 1.3840216299823802, 0.7322759674330881 }},
      {{ -0.7597171548555842, -1.2571169773685882, -0.32518437620636936, 1.0212806356673838, 0.9165371224616725, -0.19250317971610814 }},
      {{ -2.3497882574965994, 0.23878289300722322, -2.5867259166595944, 0.8432052252434499, -0.7374592701571068, 0.6061228206232958 }},
      {{ 0.3412638507438349, 0.009127030753615727, -0.43271581733053577, 0.3058216852138156, 0.4132978840654225, 0.08892908864656021 }},
      {{ -0.3843556431761765, -0.5398088470059381, -2.0478454682095735, -1.9041927205327738, -1.0368295384919808, -0.1653666006655781 }},
      {{ 0.9415170642828504, -0.4671602009419241, -2.594401365132767, 0.5011818371933664, 2.6743454901058725, 1.090931094328555 }},
      {{ -2.030867525769208, 0.9360281369657524, -2.179490537456837, 0.6315631977398317, -0.2797813498393135, 1.1780045163240112 }},
      {{ 2.026780502536945, 1.1382782700184098, 0.7076892737809293, -0.5003242829913847, 1.7337823655903326, 0.676979521067241 }},
    }},
    /*biases=*/{
      { 2.827760670625431, -0.9347274494671962, 1.7748650815163647, -0.5102747570142624, 1.1443725632238269, 2.0573020231014616, 0.33721201132380757, 2.7437956980307643 },
    }
  },
  /*hidden_layer_2=*/{
    /*weights=*/{{
      {{ 2.571821311709108, 0.16869445337763503, 0.3541411973512104, 0.31040383433531593, -1.9138308971941267, 1.577267326066108, 1.0358680188904088, -0.48597239908310547 }},
      {{ -0.3168524372865204, -0.8109707535168992, -0.6883758912881943, 0.20041683878416458, 0.29562419861502953, 2.9699371941875183, -0.06378706528945598, -1.2627270412739198 }},
      {{ 1.2121865841893051, 0.4324679330555888, 0.5756742637802713, -0.3965637421226802, -0.8316876650525071, 1.4267737797853521, 0.6590628275882154, 1.0969896994507335 }},
      {{ 0.08152092107879703, 0.987281670566132, 2.711801967605775, 0.03262333498333622, -0.24851434369301018, 0.5857580261361529, -0.14172228489696118, 1.0096244465236095 }},
      {{ -1.099617291565094, -0.96182176932886, 1.1198642662894356, 0.09569259551658717, 0.9865508260397995, -1.7073686127591108, 0.8545686868857858, 1.276785903326864 }},
      {{ 0.6284115174399925, -0.5692706408214737, -0.3776497427936689, 0.2850473804130665, 0.5611912673866001, 0.7074167980672433, 1.3602397130866593, -2.4641849404042104 }},
      {{ -0.2235255127724266, -0.6066818030776572, 2.098453748102861, -0.551860833640914, -0.6607678541967575, -1.0968858307838945, -3.097129404864497, 1.22936241411423 }},
      {{ -0.35359032516179434, 0.16659401401800453, 0.7409562527506246, 0.12880569714035928, 1.6235584538175323, 0.35055754805485, -0.5085408039033421, 0.03832167245213557 }},
    }},
    /*biases=*/{
      { -0.9650088973529635, 0.18404512445819377, -1.1301082618712814, -0.4114680200097482, -2.16829227705252, -0.792693003568079, 2.0186809343196432, 0.6651750830570318 },
    }
  },
  /*output_layer=*/{
    /*weights=*/{{
      {{ -3.4950798141841886, 3.052869401349734, -1.9332425183341917, -2.4468455334890375, 3.1182134156177734, 2.662143418701658, 3.609609051057281, -1.6114776062537006 }},
    }},
    /*biases=*/{
      { -0.8627209596023582 },
    }
  },
  /*threshold=*/0.03,
};

static constexpr GemmFilter AMDGenoaGemmFilter{
  /*input_range=*/{16, 4096},
  /*lhs_dtype=*/PrimitiveType::F32,
  /*rhs_dtype=*/PrimitiveType::F32,
  /*out_dtype=*/PrimitiveType::F32,
};

static constexpr Net AMDGenoaNet {
  /*scaler=*/{
    /*mean=*/
    {{ 2048.487742594484, 2032.4805924412667, 2042.0275791624106, 7.311636506981553, 7.331182177414692, 7.324348610024091 }},
    /*scale=*/
    {{ 1191.317145630777, 1166.4230415375375, 1162.7572402044934, 1.0130577584567735, 0.9372130582909888, 0.9819331632142719 }},
  },
  /*hidden_layer_1=*/{
    /*weights=*/{{
      {{ -0.3975566315544443, 0.5914998393825349, 0.6099048505253704, -2.2657754130482575, 0.36614796953745665, -0.9019941522654611 }},
      {{ -1.634528631004246, -1.0247790097319367, 0.7441596497436759, 1.1627072134985457, 0.05409335988074912, -0.12091065051829138 }},
      {{ 0.38395072299848293, 0.6541884828037803, 0.417837898603066, -0.9405446354332785, 2.184810649384631, -0.36876630139170674 }},
      {{ 1.4311717327837925, 0.9019482519954495, 0.010222966815173684, 0.3734603575926762, -0.48722286699557477, 0.6097423536728197 }},
      {{ -0.7136793187709407, -1.9428210404652928, 0.4274609198312262, 0.7241649472475438, 0.7127139917668667, -0.17169269406677637 }},
      {{ 0.7274093691413374, 1.5619764328746881, 0.3132760663502329, 0.1150444561729908, 0.2015964262316955, -1.6488397218364703 }},
      {{ -0.2753144111803734, 0.851664634951511, -0.7668837132534746, 0.8536953128922471, 0.5346385907475031, -0.3903852123459044 }},
      {{ -0.33049518181245935, -0.1445885038395346, 0.33671360297244707, 0.19923558301288513, 0.47714692266995923, 2.673625950077934 }},
    }},
    /*biases=*/{
      { 1.8781920773242509, 0.6510580145727756, 1.3641835181490685, -1.237083419397511, 0.09563962519162661, 1.0633713668067988, -0.2750294272946441, 0.4082406241441991 },
    }
  },
  /*hidden_layer_2=*/{
    /*weights=*/{{
      {{ 1.482788775138106, -0.5911919348052194, -0.35265948412831416, 0.5693173975201452, 0.08299331485534553, -1.0926309595949408, 0.334160671733911, -0.8259113265483281 }},
      {{ -0.7244072332431708, 1.7167578358580047, -0.4425799291591407, 0.38193961610444616, -0.3131049026459214, 0.7057668457879581, -0.8977670579096759, -1.1564071580034785 }},
      {{ 0.2358887563481682, 0.845047198622242, 0.3965633248481624, -0.9292260319808021, 0.38780851270938177, 0.9073719197977955, 0.8942857890487362, 2.2078844573893486 }},
      {{ 0.7588397006376895, 0.39649528525833017, 1.1922103753418032, -0.2623025347145879, -1.8688404509544276, 0.23950836230216038, 0.15018196046213705, 1.1091046070474726 }},
      {{ -0.06639877236719088, 0.09408482409872725, 0.08853697547037886, -0.027191640785169502, -0.025050403848262424, -0.14821218627938373, -0.05119778874800481, -0.003846457076482196 }},
      {{ -1.3626737341753659, -0.509211567650967, -1.3709529389911908, 0.8181695565961004, -0.9154056938786789, 1.6786394527771, -0.38910973671573107, 0.6109302318778375 }},
      {{ -0.9490250745418807, -0.22890259271729135, -0.7669763564967859, -1.2378100390537607, 0.9325554827865082, -0.7707072257516585, -0.6101643395959798, 0.6438447441624673 }},
      {{ 1.1581876959277013, 1.4439015663052703, -1.4659507082977212, 1.0425420146162472, -0.20891484120663645, 0.3292514803046433, 0.38947771607697135, 0.06588859566944062 }},
    }},
    /*biases=*/{
      { 2.0991435035679293, 0.9220598032166089, 0.001237522670163396, -0.2035381110666839, -0.7214610628375114, -2.275782698263265, 3.2572710355363337, -1.309956720253099 },
    }
  },
  /*output_layer=*/{
    /*weights=*/{{
      {{ -2.214950317234679, 2.3173207097966624, -2.4148863077632057, 2.440952250974181, 0.016504153668811035, 3.00219780922754, 2.454200734592688, 2.444832006369846 }},
    }},
    /*biases=*/{
      { -0.2538826384470055 },
    }
  },
  /*threshold=*/0.05,
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
