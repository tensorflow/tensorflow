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

#ifndef XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_DATA_H_
#define XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_DATA_H_

constexpr char kDefaultMatmulPTable[] = R"pb(
  entries {
    key: "sm_90"
    value {
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 139519467775467 }
        flops { key: "f32xf32->f32" value: 82342164417177 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 332247798870580 }
        flops { key: "f32xf32->f32" value: 147979854465270 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 41838443890274 }
        flops { key: "f32xf32->f32" value: 32140260536398 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 42366707070707 }
        flops { key: "f32xf32->f32" value: 31476953095684 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 238821580071174 }
        flops { key: "f32xf32->f32" value: 116711067826086 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 588674245614035 }
        flops { key: "f32xf32->f32" value: 154518848960722 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 21788592207792 }
        flops { key: "f32xf32->f32" value: 15650388059701 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 139230008298755 }
        flops { key: "f32xf32->f32" value: 62368832713754 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 493447529411764 }
        flops { key: "f32xf32->f32" value: 205860344429266 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 285873755058572 }
        flops { key: "f32xf32->f32" value: 119890779812416 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 525314003913894 }
        flops { key: "f32xf32->f32" value: 175161798368678 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 712739345502821 }
        flops { key: "f32xf32->f32" value: 284776746657826 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 343707370038412 }
        flops { key: "f32xf32->f32" value: 144709140700808 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 631984593290170 }
        flops { key: "f32xf32->f32" value: 329166714898835 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 22550021505376 }
        flops { key: "f32xf32->f32" value: 16513007874015 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 36954220264317 }
        flops { key: "f32xf32->f32" value: 21959706806282 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 91678775956284 }
        flops { key: "f32xf32->f32" value: 46442120415224 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 22671913513513 }
        flops { key: "f32xf32->f32" value: 16810837675350 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 392162828341855 }
        flops { key: "f32xf32->f32" value: 183985919122686 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 216131607085346 }
        flops { key: "f32xf32->f32" value: 82697306223043 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 678724288242730 }
        flops { key: "f32xf32->f32" value: 302806341426443 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 211034163522012 }
        flops { key: "f32xf32->f32" value: 111199443247721 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 763011394582711 }
        flops { key: "f32xf32->f32" value: 326132189596626 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 487178686025408 }
        flops { key: "f32xf32->f32" value: 206250830580099 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 514244168582375 }
        flops { key: "f32xf32->f32" value: 165649772292502 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 347714321243523 }
        flops { key: "f32xf32->f32" value: 174082656290531 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 216480206451612 }
        flops { key: "f32xf32->f32" value: 95055048158640 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 588997160724081 }
        flops { key: "f32xf32->f32" value: 229043545055794 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43690666666666 }
        flops { key: "f32xf32->f32" value: 31595510357815 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 783328793548203 }
        flops { key: "f32xf32->f32" value: 401589990188086 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 373345557719054 }
        flops { key: "f32xf32->f32" value: 162098705314009 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 119623643493761 }
        flops { key: "f32xf32->f32" value: 54515730300568 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 91056803256445 }
        flops { key: "f32xf32->f32" value: 79044598351001 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 378628050954291 }
        flops { key: "f32xf32->f32" value: 115084868595927 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 172294901155327 }
        flops { key: "f32xf32->f32" value: 73143176021798 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 447019910074937 }
        flops { key: "f32xf32->f32" value: 193886208739617 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 135847902834008 }
        flops { key: "f32xf32->f32" value: 65728564152791 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 349525333333333 }
        flops { key: "f32xf32->f32" value: 149462948775055 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 626088527113702 }
        flops { key: "f32xf32->f32" value: 217400652763717 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 673839272969739 }
        flops { key: "f32xf32->f32" value: 278534507964558 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 218240208130081 }
        flops { key: "f32xf32->f32" value: 83938541588492 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 66576253968253 }
        flops { key: "f32xf32->f32" value: 35734219382321 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 619859436746254 }
        flops { key: "f32xf32->f32" value: 319734034048770 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 523265996101364 }
        flops { key: "f32xf32->f32" value: 254561835941204 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 358870930481283 }
        flops { key: "f32xf32->f32" value: 144865329735563 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 750605958755679 }
        flops { key: "f32xf32->f32" value: 306217422870230 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 306433168949771 }
        flops { key: "f32xf32->f32" value: 102573731753916 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 78766272300469 }
        flops { key: "f32xf32->f32" value: 51781530864197 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 11459846994535 }
        flops { key: "f32xf32->f32" value: 9731563805104 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 77852510440835 }
        flops { key: "f32xf32->f32" value: 49490312684365 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 68338965376782 }
        flops { key: "f32xf32->f32" value: 39709386982248 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 304694047673098 }
        flops { key: "f32xf32->f32" value: 178718679094540 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 349981037809648 }
        flops { key: "f32xf32->f32" value: 160361695702497 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 441505684210526 }
        flops { key: "f32xf32->f32" value: 223882782318598 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 301612871910112 }
        flops { key: "f32xf32->f32" value: 162294713422007 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 11459846994535 }
        flops { key: "f32xf32->f32" value: 9731563805104 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 729692031260618 }
        flops { key: "f32xf32->f32" value: 223348685105857 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 596523235555555 }
        flops { key: "f32xf32->f32" value: 270604520358498 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 41221660933660 }
        flops { key: "f32xf32->f32" value: 25536097412480 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 324589426844014 }
        flops { key: "f32xf32->f32" value: 131521536501714 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 725746417032781 }
        flops { key: "f32xf32->f32" value: 317725034149228 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 220029062295081 }
        flops { key: "f32xf32->f32" value: 94586136715997 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 783185953785750 }
        flops { key: "f32xf32->f32" value: 320546111440999 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 542037204101593 }
        flops { key: "f32xf32->f32" value: 263236534444716 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 503874974234136 }
        flops { key: "f32xf32->f32" value: 242271975405963 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 621378370370370 }
        flops { key: "f32xf32->f32" value: 272873920869137 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 232613046793760 }
        flops { key: "f32xf32->f32" value: 111848106666666 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 370767204419889 }
        flops { key: "f32xf32->f32" value: 177771825165562 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 592245904026475 }
        flops { key: "f32xf32->f32" value: 257739276044167 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 171633923273657 }
        flops { key: "f32xf32->f32" value: 70640909473684 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 524288000000000 }
        flops { key: "f32xf32->f32" value: 224069662771285 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 600191069871436 }
        flops { key: "f32xf32->f32" value: 269314937593077 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 368224219478738 }
        flops { key: "f32xf32->f32" value: 118934628267611 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 793131337247852 }
        flops { key: "f32xf32->f32" value: 346934459512435 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 710616693580410 }
        flops { key: "f32xf32->f32" value: 292535340070665 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 394467973548861 }
        flops { key: "f32xf32->f32" value: 210125601565557 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 488064465454545 }
        flops { key: "f32xf32->f32" value: 259107583011583 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 293051807860262 }
        flops { key: "f32xf32->f32" value: 117889967501097 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 76433785876993 }
        flops { key: "f32xf32->f32" value: 50006605067064 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 481930800718132 }
        flops { key: "f32xf32->f32" value: 188843726603205 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 607491838189533 }
        flops { key: "f32xf32->f32" value: 289264780044282 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 21564544987146 }
        flops { key: "f32xf32->f32" value: 15307678832116 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 373345557719054 }
        flops { key: "f32xf32->f32" value: 174762666666666 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 393312023443223 }
        flops { key: "f32xf32->f32" value: 198473534935305 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 543116754678806 }
        flops { key: "f32xf32->f32" value: 234954447264770 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 728585722240481 }
        flops { key: "f32xf32->f32" value: 267534616782552 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 445536026556016 }
        flops { key: "f32xf32->f32" value: 229144359164510 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 717741860962566 }
        flops { key: "f32xf32->f32" value: 290910570294045 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 39107729603729 }
        flops { key: "f32xf32->f32" value: 21902370757180 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 342392163265306 }
        flops { key: "f32xf32->f32" value: 157347864009378 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 777585153532370 }
        flops { key: "f32xf32->f32" value: 310086351656483 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 138084082304526 }
        flops { key: "f32xf32->f32" value: 79324898345153 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 340654131979695 }
        flops { key: "f32xf32->f32" value: 181990139661016 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 136123456389452 }
        flops { key: "f32xf32->f32" value: 76346830489192 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 72471775377969 }
        flops { key: "f32xf32->f32" value: 40427026506024 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 135847902834008 }
        flops { key: "f32xf32->f32" value: 60567566787003 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 77314359447004 }
        flops { key: "f32xf32->f32" value: 45039506040268 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 21959706806282 }
        flops { key: "f32xf32->f32" value: 17084741344195 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 307486203894616 }
        flops { key: "f32xf32->f32" value: 118152658688894 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 458472170794193 }
        flops { key: "f32xf32->f32" value: 222398886495443 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 384045003442571 }
        flops { key: "f32xf32->f32" value: 115134229466009 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 807671014191940 }
        flops { key: "f32xf32->f32" value: 340050234669602 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 168615236180904 }
        flops { key: "f32xf32->f32" value: 70455500262467 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 546433498218829 }
        flops { key: "f32xf32->f32" value: 279765978113600 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 40329846153846 }
        flops { key: "f32xf32->f32" value: 25653235474006 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 629391456037514 }
        flops { key: "f32xf32->f32" value: 218732021746049 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 358391797062750 }
        flops { key: "f32xf32->f32" value: 143089262260127 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 216480206451612 }
        flops { key: "f32xf32->f32" value: 93727463687150 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 39290903981264 }
        flops { key: "f32xf32->f32" value: 22519753020134 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 31126560296846 }
        flops { key: "f32xf32->f32" value: 19217887743413 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 217180789644012 }
        flops { key: "f32xf32->f32" value: 86313651446945 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 142481664543524 }
        flops { key: "f32xf32->f32" value: 80854053012048 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 594212409518539 }
        flops { key: "f32xf32->f32" value: 294579375582990 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 808699906572796 }
        flops { key: "f32xf32->f32" value: 339606581835209 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 674884867379006 }
        flops { key: "f32xf32->f32" value: 297355612396257 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 738736406437117 }
        flops { key: "f32xf32->f32" value: 324960877363219 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 114716006837606 }
        flops { key: "f32xf32->f32" value: 55097589490968 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 674037554300062 }
        flops { key: "f32xf32->f32" value: 301869906503957 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 518715857004830 }
        flops { key: "f32xf32->f32" value: 168990076764179 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 91304576870748 }
        flops { key: "f32xf32->f32" value: 79044598351001 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 486737000906618 }
        flops { key: "f32xf32->f32" value: 210951242436149 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 464019802938634 }
        flops { key: "f32xf32->f32" value: 230617748627424 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 649964784503632 }
        flops { key: "f32xf32->f32" value: 315620759553204 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 216480206451612 }
        flops { key: "f32xf32->f32" value: 113073064869418 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 490293070319634 }
        flops { key: "f32xf32->f32" value: 225860711821623 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 78033562790697 }
        flops { key: "f32xf32->f32" value: 54471480519480 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 11554556473829 }
        flops { key: "f32xf32->f32" value: 9709037037037 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 629391456037514 }
        flops { key: "f32xf32->f32" value: 227873901528013 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 347714321243523 }
        flops { key: "f32xf32->f32" value: 147330107574094 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 235469698245614 }
        flops { key: "f32xf32->f32" value: 111107390728476 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 412026793553338 }
        flops { key: "f32xf32->f32" value: 177834391073018 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 738474431911967 }
        flops { key: "f32xf32->f32" value: 291068289477919 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 478494573975044 }
        flops { key: "f32xf32->f32" value: 166474826876489 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 169681072060682 }
        flops { key: "f32xf32->f32" value: 69832324661810 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 80854053012048 }
        flops { key: "f32xf32->f32" value: 50917195751138 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 21732145077720 }
        flops { key: "f32xf32->f32" value: 16677153081510 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 78215459207459 }
        flops { key: "f32xf32->f32" value: 50533783132530 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 136400130081300 }
        flops { key: "f32xf32->f32" value: 62543209692451 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 508400484848484 }
        flops { key: "f32xf32->f32" value: 158512199295085 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43577184415584 }
        flops { key: "f32xf32->f32" value: 31714964083175 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 519217516441005 }
        flops { key: "f32xf32->f32" value: 167719747578881 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 204600195121951 }
        flops { key: "f32xf32->f32" value: 82544728167281 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 132364623274161 }
        flops { key: "f32xf32->f32" value: 67513947686116 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 304003913929784 }
        flops { key: "f32xf32->f32" value: 132104062992125 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 518215166023166 }
        flops { key: "f32xf32->f32" value: 170273045353631 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 561874319204604 }
        flops { key: "f32xf32->f32" value: 170869163590070 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 226336809443507 }
        flops { key: "f32xf32->f32" value: 110649404781533 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 607674484339352 }
        flops { key: "f32xf32->f32" value: 208697443303227 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 598184860167130 }
        flops { key: "f32xf32->f32" value: 264863931424695 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 40920039024390 }
        flops { key: "f32xf32->f32" value: 25970922600619 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 21959706806282 }
        flops { key: "f32xf32->f32" value: 15592208178438 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 327760019536019 }
        flops { key: "f32xf32->f32" value: 147330107574094 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 758434522012648 }
        flops { key: "f32xf32->f32" value: 384029980138926 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 764575547036718 }
        flops { key: "f32xf32->f32" value: 342271935609352 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 71392408510638 }
        flops { key: "f32xf32->f32" value: 37532921700223 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 371280022130013 }
        flops { key: "f32xf32->f32" value: 144865329735563 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 598184860167130 }
        flops { key: "f32xf32->f32" value: 266971285356870 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 493901482980680 }
        flops { key: "f32xf32->f32" value: 250640014939309 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 487178686025408 }
        flops { key: "f32xf32->f32" value: 197016848440366 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 41838443890274 }
        flops { key: "f32xf32->f32" value: 25930782071097 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 688634012446012 }
        flops { key: "f32xf32->f32" value: 325083857968683 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 343267846547314 }
        flops { key: "f32xf32->f32" value: 147898322865013 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 298593388209121 }
        flops { key: "f32xf32->f32" value: 161708106024096 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 631241519106408 }
        flops { key: "f32xf32->f32" value: 232113344376139 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 77492914549653 }
        flops { key: "f32xf32->f32" value: 51861564142194 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 145572373101952 }
        flops { key: "f32xf32->f32" value: 80369897005988 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 703631601572739 }
        flops { key: "f32xf32->f32" value: 244339392332690 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 699734000651678 }
        flops { key: "f32xf32->f32" value: 276276973537996 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 235883528998242 }
        flops { key: "f32xf32->f32" value: 107288351718625 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 139810133333333 }
        flops { key: "f32xf32->f32" value: 68548379979570 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 371794260387811 }
        flops { key: "f32xf32->f32" value: 135163875125881 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 138654677685950 }
        flops { key: "f32xf32->f32" value: 66182311637080 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 228261442176870 }
        flops { key: "f32xf32->f32" value: 111662003327787 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 680876235890932 }
        flops { key: "f32xf32->f32" value: 276028393287221 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 430530001603849 }
        flops { key: "f32xf32->f32" value: 181252839972991 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 458864027350427 }
        flops { key: "f32xf32->f32" value: 221164911803705 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 756697425931839 }
        flops { key: "f32xf32->f32" value: 301108029620282 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 373345557719054 }
        flops { key: "f32xf32->f32" value: 186932768802228 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 11459846994535 }
        flops { key: "f32xf32->f32" value: 9218250549450 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 720518760010484 }
        flops { key: "f32xf32->f32" value: 298160677964925 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 450395060402684 }
        flops { key: "f32xf32->f32" value: 192289008595988 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 489845722627737 }
        flops { key: "f32xf32->f32" value: 198331476807277 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 602887043234138 }
        flops { key: "f32xf32->f32" value: 205855411042944 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 801942750700917 }
        flops { key: "f32xf32->f32" value: 323686992182136 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 82241254901960 }
        flops { key: "f32xf32->f32" value: 45221606469002 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43129089974293 }
        flops { key: "f32xf32->f32" value: 31595510357815 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 335544320000000 }
        flops { key: "f32xf32->f32" value: 168403673776662 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 477643160142348 }
        flops { key: "f32xf32->f32" value: 158743616794795 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 299593142857142 }
        flops { key: "f32xf32->f32" value: 161805579264617 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 235057316987740 }
        flops { key: "f32xf32->f32" value: 110923742148760 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 542293850505050 }
        flops { key: "f32xf32->f32" value: 267632558325024 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 72944417391304 }
        flops { key: "f32xf32->f32" value: 37449142857142 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 541746631685166 }
        flops { key: "f32xf32->f32" value: 265518896867224 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 112598765100671 }
        flops { key: "f32xf32->f32" value: 53303307386814 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 237553500884955 }
        flops { key: "f32xf32->f32" value: 97612893090909 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 79324898345153 }
        flops { key: "f32xf32->f32" value: 54120051612903 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 313959597660818 }
        flops { key: "f32xf32->f32" value: 102888254503641 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 216131607085346 }
        flops { key: "f32xf32->f32" value: 96006958512160 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 335544320000000 }
        flops { key: "f32xf32->f32" value: 176486164365548 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 738474431911967 }
        flops { key: "f32xf32->f32" value: 324837989770739 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 84307618090452 }
        flops { key: "f32xf32->f32" value: 59178892416225 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 135027895372233 }
        flops { key: "f32xf32->f32" value: 75318590347923 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 309614136101499 }
        flops { key: "f32xf32->f32" value: 120590950584007 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 737206882251973 }
        flops { key: "f32xf32->f32" value: 287329623508358 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 759301869386266 }
        flops { key: "f32xf32->f32" value: 293054307299973 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 443328581337737 }
        flops { key: "f32xf32->f32" value: 226910782755705 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81442796116504 }
        flops { key: "f32xf32->f32" value: 51463852760736 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 60349697841726 }
        flops { key: "f32xf32->f32" value: 28102539363484 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 174535407022106 }
        flops { key: "f32xf32->f32" value: 71851032119914 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43464290155440 }
        flops { key: "f32xf32->f32" value: 32704124756335 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 645277538461538 }
        flops { key: "f32xf32->f32" value: 180369867965731 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 714636821297837 }
        flops { key: "f32xf32->f32" value: 287946050499886 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 215784128617363 }
        flops { key: "f32xf32->f32" value: 111107390728476 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 209715200000000 }
        flops { key: "f32xf32->f32" value: 106184911392405 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 587707621237000 }
        flops { key: "f32xf32->f32" value: 177157712647589 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 140395112970711 }
        flops { key: "f32xf32->f32" value: 80177854241338 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 521740439261418 }
        flops { key: "f32xf32->f32" value: 259734355104015 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 737475872335858 }
        flops { key: "f32xf32->f32" value: 250700363853927 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 225197530201342 }
        flops { key: "f32xf32->f32" value: 101449529856387 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 206806976887519 }
        flops { key: "f32xf32->f32" value: 110558260296540 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 487178686025408 }
        flops { key: "f32xf32->f32" value: 230819147977965 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 692967184002742 }
        flops { key: "f32xf32->f32" value: 312069047782530 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 142179796610169 }
        flops { key: "f32xf32->f32" value: 75234152466367 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 729080438554983 }
        flops { key: "f32xf32->f32" value: 316974680283028 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 21620123711340 }
        flops { key: "f32xf32->f32" value: 16912516129032 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 141879205073995 }
        flops { key: "f32xf32->f32" value: 80466263788968 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 217180789644012 }
        flops { key: "f32xf32->f32" value: 106017162717219 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 522247968871595 }
        flops { key: "f32xf32->f32" value: 225292031892572 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 391305329446064 }
        flops { key: "f32xf32->f32" value: 196081414170927 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 140395112970711 }
        flops { key: "f32xf32->f32" value: 80562861944777 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 631984593290170 }
        flops { key: "f32xf32->f32" value: 335859187988739 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 449640629815745 }
        flops { key: "f32xf32->f32" value: 179195898531375 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 442233041186161 }
        flops { key: "f32xf32->f32" value: 223144163969346 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 209061881619937 }
        flops { key: "f32xf32->f32" value: 111107390728476 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 142481664543524 }
        flops { key: "f32xf32->f32" value: 79324898345153 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 588029476451259 }
        flops { key: "f32xf32->f32" value: 174876518566775 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 299259148272017 }
        flops { key: "f32xf32->f32" value: 143548372192513 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 753643516181744 }
        flops { key: "f32xf32->f32" value: 383517745845006 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 607320036199095 }
        flops { key: "f32xf32->f32" value: 295554929835275 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 649571581367211 }
        flops { key: "f32xf32->f32" value: 180796956357934 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 596543948887114 }
        flops { key: "f32xf32->f32" value: 260048879631872 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 734684792336640 }
        flops { key: "f32xf32->f32" value: 251847002279540 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 6452775384615 }
        flops { key: "f32xf32->f32" value: 5652700808625 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 235469698245614 }
        flops { key: "f32xf32->f32" value: 110376421052631 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 236298816901408 }
        flops { key: "f32xf32->f32" value: 107805404016064 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 113168404721753 }
        flops { key: "f32xf32->f32" value: 55692003319502 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 142784817021276 }
        flops { key: "f32xf32->f32" value: 80562861944777 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 338933656565656 }
        flops { key: "f32xf32->f32" value: 149546215041782 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 81049352657004 }
        flops { key: "f32xf32->f32" value: 50006605067064 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 286483944503735 }
        flops { key: "f32xf32->f32" value: 94552820007044 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 80659692307692 }
        flops { key: "f32xf32->f32" value: 48841967976710 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81442796116504 }
        flops { key: "f32xf32->f32" value: 51228140458015 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 641806230723251 }
        flops { key: "f32xf32->f32" value: 184745668272539 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 345476777348777 }
        flops { key: "f32xf32->f32" value: 147735528893780 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 493901482980680 }
        flops { key: "f32xf32->f32" value: 155299656349435 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 234646377622377 }
        flops { key: "f32xf32->f32" value: 110376421052631 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 211699886435331 }
        flops { key: "f32xf32->f32" value: 113168404721753 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 773528256013687 }
        flops { key: "f32xf32->f32" value: 315414325188586 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 236715569664903 }
        flops { key: "f32xf32->f32" value: 110200833786626 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 238397385435168 }
        flops { key: "f32xf32->f32" value: 124506241187384 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 141281818947368 }
        flops { key: "f32xf32->f32" value: 68548379979570 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 584190328618063 }
        flops { key: "f32xf32->f32" value: 171414722860791 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 337654661635220 }
        flops { key: "f32xf32->f32" value: 176486164365548 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 141281818947368 }
        flops { key: "f32xf32->f32" value: 76000978482446 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 430530001603849 }
        flops { key: "f32xf32->f32" value: 176196557925828 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 716066571523841 }
        flops { key: "f32xf32->f32" value: 311322572580572 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 138084082304526 }
        flops { key: "f32xf32->f32" value: 67650064516129 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 139230008298755 }
        flops { key: "f32xf32->f32" value: 80466263788968 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 442233041186161 }
        flops { key: "f32xf32->f32" value: 178243994687915 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 738998566899666 }
        flops { key: "f32xf32->f32" value: 285990347902083 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 42048160401002 }
        flops { key: "f32xf32->f32" value: 25653235474006 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 206488812307692 }
        flops { key: "f32xf32->f32" value: 103883690402476 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 730064133265340 }
        flops { key: "f32xf32->f32" value: 307093213402808 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 418776062402496 }
        flops { key: "f32xf32->f32" value: 179076354903268 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 542293850505050 }
        flops { key: "f32xf32->f32" value: 274965896030729 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 138654677685950 }
        flops { key: "f32xf32->f32" value: 67581937562940 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 146206675381263 }
        flops { key: "f32xf32->f32" value: 75743638826185 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 236298816901408 }
        flops { key: "f32xf32->f32" value: 114033753610875 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 300284366636370 }
        flops { key: "f32xf32->f32" value: 158182354743665 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 488953471766848 }
        flops { key: "f32xf32->f32" value: 200180247302556 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 138654677685950 }
        flops { key: "f32xf32->f32" value: 81541754556500 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 336385283208020 }
        flops { key: "f32xf32->f32" value: 171086970044614 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 213044012698412 }
        flops { key: "f32xf32->f32" value: 113551377326565 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 622459028405797 }
        flops { key: "f32xf32->f32" value: 167969819797711 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 238397385435168 }
        flops { key: "f32xf32->f32" value: 111015490488006 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 213382715421303 }
        flops { key: "f32xf32->f32" value: 111199443247721 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81840078048780 }
        flops { key: "f32xf32->f32" value: 51385041347626 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 71240832271762 }
        flops { key: "f32xf32->f32" value: 38971465737514 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 664033286332714 }
        flops { key: "f32xf32->f32" value: 230393525114828 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 752455207507089 }
        flops { key: "f32xf32->f32" value: 329143854183181 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 215784128617363 }
        flops { key: "f32xf32->f32" value: 95257436479772 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 737966889347079 }
        flops { key: "f32xf32->f32" value: 366781651896369 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 294984017582417 }
        flops { key: "f32xf32->f32" value: 152954675783475 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 371794260387811 }
        flops { key: "f32xf32->f32" value: 137871317925012 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 593227527071823 }
        flops { key: "f32xf32->f32" value: 296941876106194 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 644116271145770 }
        flops { key: "f32xf32->f32" value: 189975552724699 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 725508891943538 }
        flops { key: "f32xf32->f32" value: 323102203176976 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 228650303236797 }
        flops { key: "f32xf32->f32" value: 110285725554642 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 743588520775623 }
        flops { key: "f32xf32->f32" value: 298057220898862 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 295308532453245 }
        flops { key: "f32xf32->f32" value: 95088719801629 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 62368832713754 }
        flops { key: "f32xf32->f32" value: 28605653878942 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 42908480818414 }
        flops { key: "f32xf32->f32" value: 32388447876447 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 451152026890756 }
        flops { key: "f32xf32->f32" value: 165496581997533 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 236298816901408 }
        flops { key: "f32xf32->f32" value: 111107390728476 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 634224349675132 }
        flops { key: "f32xf32->f32" value: 269108226566416 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 141579881856540 }
        flops { key: "f32xf32->f32" value: 81640953771289 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 642190086124401 }
        flops { key: "f32xf32->f32" value: 177624784780810 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 214748364800000 }
        flops { key: "f32xf32->f32" value: 110832145334434 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 298261617777777 }
        flops { key: "f32xf32->f32" value: 145893790414076 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 60133390681003 }
        flops { key: "f32xf32->f32" value: 28556963404255 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 312134251162790 }
        flops { key: "f32xf32->f32" value: 118570170775474 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 309971658198614 }
        flops { key: "f32xf32->f32" value: 104857600000000 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 339791716455696 }
        flops { key: "f32xf32->f32" value: 183608383036935 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 175218966057441 }
        flops { key: "f32xf32->f32" value: 69977960375391 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43018502564102 }
        flops { key: "f32xf32->f32" value: 31536120300751 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 335125413233458 }
        flops { key: "f32xf32->f32" value: 158182354743665 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 19463127610208 }
        flops { key: "f32xf32->f32" value: 14315030716723 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 747739211298869 }
        flops { key: "f32xf32->f32" value: 298907694304529 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 653127630170316 }
        flops { key: "f32xf32->f32" value: 307927107542300 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 82241254901960 }
        flops { key: "f32xf32->f32" value: 54120051612903 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 461229305841924 }
        flops { key: "f32xf32->f32" value: 180947391978429 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 142179796610169 }
        flops { key: "f32xf32->f32" value: 75658245772266 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 80659692307692 }
        flops { key: "f32xf32->f32" value: 49344752941176 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 140102012526096 }
        flops { key: "f32xf32->f32" value: 76520939566704 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 366214810368349 }
        flops { key: "f32xf32->f32" value: 136469474326385 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 82040176039119 }
        flops { key: "f32xf32->f32" value: 51542906298003 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 735943676490747 }
        flops { key: "f32xf32->f32" value: 237765556725785 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 118987347517730 }
        flops { key: "f32xf32->f32" value: 56775688663282 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 713449716943521 }
        flops { key: "f32xf32->f32" value: 286677555112427 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 143395008547008 }
        flops { key: "f32xf32->f32" value: 79796508917954 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 394758023529411 }
        flops { key: "f32xf32->f32" value: 200999967053538 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 447019910074937 }
        flops { key: "f32xf32->f32" value: 230614652920962 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 462819751724137 }
        flops { key: "f32xf32->f32" value: 165140237465395 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 442233041186161 }
        flops { key: "f32xf32->f32" value: 216305766317485 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 376223484232655 }
        flops { key: "f32xf32->f32" value: 216305766317485 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 209715200000000 }
        flops { key: "f32xf32->f32" value: 86037005128205 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 22310127659574 }
        flops { key: "f32xf32->f32" value: 16545577909270 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 135027895372233 }
        flops { key: "f32xf32->f32" value: 65600062561094 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 644116271145770 }
        flops { key: "f32xf32->f32" value: 264212189286867 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 375434204195804 }
        flops { key: "f32xf32->f32" value: 150468304932735 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 470114633975481 }
        flops { key: "f32xf32->f32" value: 181620741542625 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 779348875095689 }
        flops { key: "f32xf32->f32" value: 320786271886174 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 622820083526682 }
        flops { key: "f32xf32->f32" value: 205542624504983 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 779136806172370 }
        flops { key: "f32xf32->f32" value: 321542765401136 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 134217728000000 }
        flops { key: "f32xf32->f32" value: 69542864248704 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 677439636593059 }
        flops { key: "f32xf32->f32" value: 309037697921444 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 718702693440428 }
        flops { key: "f32xf32->f32" value: 311503208144837 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 83886080000000 }
        flops { key: "f32xf32->f32" value: 54648912052117 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 237133795053003 }
        flops { key: "f32xf32->f32" value: 110837865703225 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 228261442176870 }
        flops { key: "f32xf32->f32" value: 110740699669967 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 335125413233458 }
        flops { key: "f32xf32->f32" value: 168087323731997 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 627185644859813 }
        flops { key: "f32xf32->f32" value: 233930680610021 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 730436614965986 }
        flops { key: "f32xf32->f32" value: 298181377997240 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 687861993493688 }
        flops { key: "f32xf32->f32" value: 298202080903986 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 237553500884955 }
        flops { key: "f32xf32->f32" value: 108854605028386 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 82040176039119 }
        flops { key: "f32xf32->f32" value: 54032901771336 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 300936609865470 }
        flops { key: "f32xf32->f32" value: 162294713422007 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 134486701402805 }
        flops { key: "f32xf32->f32" value: 79986727056019 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 342392163265306 }
        flops { key: "f32xf32->f32" value: 167982137672090 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 733304984804507 }
        flops { key: "f32xf32->f32" value: 275585610792515 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 117734849122807 }
        flops { key: "f32xf32->f32" value: 54920046238044 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 393889150403521 }
        flops { key: "f32xf32->f32" value: 202669275953189 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 298925897550111 }
        flops { key: "f32xf32->f32" value: 118462248896734 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43351979328165 }
        flops { key: "f32xf32->f32" value: 32201950095969 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 80466263788968 }
        flops { key: "f32xf32->f32" value: 50686453172205 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 715350982011992 }
        flops { key: "f32xf32->f32" value: 316600830834723 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 653525151552039 }
        flops { key: "f32xf32->f32" value: 266503307024075 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 445906073089701 }
        flops { key: "f32xf32->f32" value: 226719135135135 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 137236940695296 }
        flops { key: "f32xf32->f32" value: 78398205607476 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 373865537604456 }
        flops { key: "f32xf32->f32" value: 136608374554707 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 40721398058252 }
        flops { key: "f32xf32->f32" value: 25970922600619 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 234236872600349 }
        flops { key: "f32xf32->f32" value: 111107390728476 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 745015413610295 }
        flops { key: "f32xf32->f32" value: 302422553078378 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 206488812307692 }
        flops { key: "f32xf32->f32" value: 110014531147540 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 210372614420062 }
        flops { key: "f32xf32->f32" value: 94719638673253 }
      }
    }
  }
)pb";

#endif  // XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_DATA_H_
