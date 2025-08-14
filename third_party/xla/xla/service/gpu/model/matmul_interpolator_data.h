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
  entries {
    key: "sm_100"
    value {
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 437815218756371 }
        flops { key: "f32xf32->f32" value: 273861333673404 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 369650339616145 }
        flops { key: "f32xf32->f32" value: 249663854909027 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 454878976488032 }
        flops { key: "f32xf32->f32" value: 290475266874070 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 426003500892680 }
        flops { key: "f32xf32->f32" value: 293011822622458 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 678617047874861 }
        flops { key: "f32xf32->f32" value: 406643372088619 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 74051160275862 }
        flops { key: "f32xf32->f32" value: 54637788723794 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 638229778735418 }
        flops { key: "f32xf32->f32" value: 407241008486227 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 434229834799312 }
        flops { key: "f32xf32->f32" value: 273303677760101 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1543980334681405 }
        flops { key: "f32xf32->f32" value: 845590844317566 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 368192652893270 }
        flops { key: "f32xf32->f32" value: 264663994084298 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1276483268059812 }
        flops { key: "f32xf32->f32" value: 730390033968922 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 103393531439576 }
        flops { key: "f32xf32->f32" value: 67501214811089 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 497016408725337 }
        flops { key: "f32xf32->f32" value: 344285955591182 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 619139007640190 }
        flops { key: "f32xf32->f32" value: 393510219982591 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 40414853357422 }
        flops { key: "f32xf32->f32" value: 24704164918093 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 83235800310077 }
        flops { key: "f32xf32->f32" value: 49922904221684 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 518590593576430 }
        flops { key: "f32xf32->f32" value: 349952521469893 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1141633331162574 }
        flops { key: "f32xf32->f32" value: 626705183088315 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 418001683309002 }
        flops { key: "f32xf32->f32" value: 286426628609536 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 465882123440720 }
        flops { key: "f32xf32->f32" value: 299530462096380 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 9216984480153 }
        flops { key: "f32xf32->f32" value: 10510393735317 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 898740246606157 }
        flops { key: "f32xf32->f32" value: 610644387005047 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 250844953626912 }
        flops { key: "f32xf32->f32" value: 165867277979454 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 868679232643980 }
        flops { key: "f32xf32->f32" value: 512434205810415 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 11846224889673 }
        flops { key: "f32xf32->f32" value: 10951185378590 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 258546068865880 }
        flops { key: "f32xf32->f32" value: 169882418163120 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 616296067728512 }
        flops { key: "f32xf32->f32" value: 393240001464933 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 81037118792452 }
        flops { key: "f32xf32->f32" value: 57247911281723 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 42683328987120 }
        flops { key: "f32xf32->f32" value: 30608375826681 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1324942674121775 }
        flops { key: "f32xf32->f32" value: 703898273387485 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 192824247822573 }
        flops { key: "f32xf32->f32" value: 131432991492747 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 271661435547122 }
        flops { key: "f32xf32->f32" value: 186364978564609 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1462661134166977 }
        flops { key: "f32xf32->f32" value: 873213190244894 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 656171002368039 }
        flops { key: "f32xf32->f32" value: 398789906778087 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 941723904182426 }
        flops { key: "f32xf32->f32" value: 570418659406335 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1394455753005752 }
        flops { key: "f32xf32->f32" value: 783958802331816 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1185758993960727 }
        flops { key: "f32xf32->f32" value: 653985389291764 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 710969590465154 }
        flops { key: "f32xf32->f32" value: 407569490985006 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 768467936303453 }
        flops { key: "f32xf32->f32" value: 455506129600169 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1190854967178456 }
        flops { key: "f32xf32->f32" value: 645132151107773 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 871322675051985 }
        flops { key: "f32xf32->f32" value: 511686349486224 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 432873140092723 }
        flops { key: "f32xf32->f32" value: 272471439193047 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 239193990643795 }
        flops { key: "f32xf32->f32" value: 170516408448467 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 126352297481760 }
        flops { key: "f32xf32->f32" value: 104175979819540 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 22482031490787 }
        flops { key: "f32xf32->f32" value: 17327359669506 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 256110154800238 }
        flops { key: "f32xf32->f32" value: 174720010414124 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43240247422680 }
        flops { key: "f32xf32->f32" value: 32443250664732 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 442871447308723 }
        flops { key: "f32xf32->f32" value: 280735165435649 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1329235407913189 }
        flops { key: "f32xf32->f32" value: 813185692650861 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 149462948775055 }
        flops { key: "f32xf32->f32" value: 99540356354871 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 241344532254439 }
        flops { key: "f32xf32->f32" value: 160511521638388 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 466742805477070 }
        flops { key: "f32xf32->f32" value: 297538434083824 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 620525506898793 }
        flops { key: "f32xf32->f32" value: 386446580529062 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 696161325228948 }
        flops { key: "f32xf32->f32" value: 422670599419377 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1337930917225602 }
        flops { key: "f32xf32->f32" value: 814834520883850 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1048000316232538 }
        flops { key: "f32xf32->f32" value: 609939793158539 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 133152507936507 }
        flops { key: "f32xf32->f32" value: 89590473425114 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 195599202841788 }
        flops { key: "f32xf32->f32" value: 130912195074372 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 679583432911392 }
        flops { key: "f32xf32->f32" value: 404175156072083 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1270066289685253 }
        flops { key: "f32xf32->f32" value: 740080952204536 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 370671208768447 }
        flops { key: "f32xf32->f32" value: 229812579378243 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 20757458707083 }
        flops { key: "f32xf32->f32" value: 16008793893129 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 22306419810536 }
        flops { key: "f32xf32->f32" value: 17655581162851 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 769018316204118 }
        flops { key: "f32xf32->f32" value: 453737664316086 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 764059114253947 }
        flops { key: "f32xf32->f32" value: 504089351368797 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 196188895304220 }
        flops { key: "f32xf32->f32" value: 132071565067650 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 772302503214205 }
        flops { key: "f32xf32->f32" value: 496772089870745 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 258546068865880 }
        flops { key: "f32xf32->f32" value: 158077559661391 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 380658273154302 }
        flops { key: "f32xf32->f32" value: 254185198319228 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 23168950112204 }
        flops { key: "f32xf32->f32" value: 16508945633456 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 623407692285361 }
        flops { key: "f32xf32->f32" value: 378878554693013 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 442187511170596 }
        flops { key: "f32xf32->f32" value: 293974489801505 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1339613176654060 }
        flops { key: "f32xf32->f32" value: 674540389650163 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 239193990643795 }
        flops { key: "f32xf32->f32" value: 167523492316093 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 371762078767419 }
        flops { key: "f32xf32->f32" value: 263656678698588 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1411121014733513 }
        flops { key: "f32xf32->f32" value: 748508871575072 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1416706559656953 }
        flops { key: "f32xf32->f32" value: 754906067043463 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 685549448683160 }
        flops { key: "f32xf32->f32" value: 401155120347452 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 37866477077161 }
        flops { key: "f32xf32->f32" value: 26087021963070 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 382863905865573 }
        flops { key: "f32xf32->f32" value: 252022491256894 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 429410847430513 }
        flops { key: "f32xf32->f32" value: 273025700591189 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 21672489585015 }
        flops { key: "f32xf32->f32" value: 14923029575272 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 270532079617032 }
        flops { key: "f32xf32->f32" value: 182593627072527 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 423358038048299 }
        flops { key: "f32xf32->f32" value: 276666277763463 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 100134460878485 }
        flops { key: "f32xf32->f32" value: 68048787882628 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 872783437512700 }
        flops { key: "f32xf32->f32" value: 524192017574906 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 447299239325140 }
        flops { key: "f32xf32->f32" value: 276399208185854 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1186864883177893 }
        flops { key: "f32xf32->f32" value: 657413916923371 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 611339733257419 }
        flops { key: "f32xf32->f32" value: 390433825371574 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 516594574933846 }
        flops { key: "f32xf32->f32" value: 350152233490950 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 423274593081699 }
        flops { key: "f32xf32->f32" value: 298199492883427 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1690656401017304 }
        flops { key: "f32xf32->f32" value: 872390099668027 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 265186916275623 }
        flops { key: "f32xf32->f32" value: 189288994975760 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1338569417117924 }
        flops { key: "f32xf32->f32" value: 705104419618305 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 80841878030417 }
        flops { key: "f32xf32->f32" value: 59061706490649 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 625586963221906 }
        flops { key: "f32xf32->f32" value: 424382915468603 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1320056028583503 }
        flops { key: "f32xf32->f32" value: 687731197694201 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1411845803897398 }
        flops { key: "f32xf32->f32" value: 754591068657109 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1341469864251273 }
        flops { key: "f32xf32->f32" value: 693743707963172 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 44027465310808 }
        flops { key: "f32xf32->f32" value: 29070333116742 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 868503573327941 }
        flops { key: "f32xf32->f32" value: 517372438234054 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 81024888620585 }
        flops { key: "f32xf32->f32" value: 55450414377194 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 78951604705882 }
        flops { key: "f32xf32->f32" value: 53419991243781 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 433616082382635 }
        flops { key: "f32xf32->f32" value: 272177902154626 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 434273740748230 }
        flops { key: "f32xf32->f32" value: 264925197137922 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 413494492731298 }
        flops { key: "f32xf32->f32" value: 269716609897010 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 663520360883670 }
        flops { key: "f32xf32->f32" value: 401173855408182 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 107892064308681 }
        flops { key: "f32xf32->f32" value: 103224555277831 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81221015431164 }
        flops { key: "f32xf32->f32" value: 58153261698440 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 682878972255346 }
        flops { key: "f32xf32->f32" value: 422649802794725 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 937561077493997 }
        flops { key: "f32xf32->f32" value: 520176497532322 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 133666354288559 }
        flops { key: "f32xf32->f32" value: 88054930621617 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 517684239860182 }
        flops { key: "f32xf32->f32" value: 351283465914202 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 954225126860697 }
        flops { key: "f32xf32->f32" value: 562702472372342 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 613435306148682 }
        flops { key: "f32xf32->f32" value: 387597445717895 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1338986725692685 }
        flops { key: "f32xf32->f32" value: 703703655108853 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 42246688070506 }
        flops { key: "f32xf32->f32" value: 32764000488221 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 81827604328608 }
        flops { key: "f32xf32->f32" value: 55273438896324 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 424655655131500 }
        flops { key: "f32xf32->f32" value: 294619789820277 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 616960036773683 }
        flops { key: "f32xf32->f32" value: 388948815576182 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 442187511170596 }
        flops { key: "f32xf32->f32" value: 299196607175200 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 195902540412333 }
        flops { key: "f32xf32->f32" value: 131690908689519 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1131259288446976 }
        flops { key: "f32xf32->f32" value: 630916973338229 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 20510044009779 }
        flops { key: "f32xf32->f32" value: 16976692132557 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 44020245326336 }
        flops { key: "f32xf32->f32" value: 27817145699481 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 510212318365407 }
        flops { key: "f32xf32->f32" value: 348306487389506 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 152498483738105 }
        flops { key: "f32xf32->f32" value: 106680757476403 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1338543343968523 }
        flops { key: "f32xf32->f32" value: 815037469664174 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 1092363202975726 }
        flops { key: "f32xf32->f32" value: 668763641402935 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1039220227081632 }
        flops { key: "f32xf32->f32" value: 610232273079245 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 383959171821920 }
        flops { key: "f32xf32->f32" value: 253181283659514 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 766889973395232 }
        flops { key: "f32xf32->f32" value: 500724837773243 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 103066022653100 }
        flops { key: "f32xf32->f32" value: 67096283446853 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 959394046127212 }
        flops { key: "f32xf32->f32" value: 548002206826156 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 257553807627728 }
        flops { key: "f32xf32->f32" value: 170286547299976 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 855400775941047 }
        flops { key: "f32xf32->f32" value: 520381328648452 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 379112657427840 }
        flops { key: "f32xf32->f32" value: 252007703808015 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 674355047260166 }
        flops { key: "f32xf32->f32" value: 400556520960596 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 271661435547122 }
        flops { key: "f32xf32->f32" value: 181084716080613 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 774565788277727 }
        flops { key: "f32xf32->f32" value: 455905028368229 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 252259326676847 }
        flops { key: "f32xf32->f32" value: 163269493499581 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 74544697584004 }
        flops { key: "f32xf32->f32" value: 53762358501902 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 149775676384432 }
        flops { key: "f32xf32->f32" value: 106670159348301 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 931865327836841 }
        flops { key: "f32xf32->f32" value: 541081200088186 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 678563440398135 }
        flops { key: "f32xf32->f32" value: 403018419442619 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1172607274861784 }
        flops { key: "f32xf32->f32" value: 659394687341675 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 248063260713873 }
        flops { key: "f32xf32->f32" value: 169426717790927 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 38211452811387 }
        flops { key: "f32xf32->f32" value: 27588433299075 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 258546068865880 }
        flops { key: "f32xf32->f32" value: 184080545859763 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1455736066093293 }
        flops { key: "f32xf32->f32" value: 863592087063613 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1458447135366869 }
        flops { key: "f32xf32->f32" value: 871969454646965 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 22129880956306 }
        flops { key: "f32xf32->f32" value: 17115242030094 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 421281735752820 }
        flops { key: "f32xf32->f32" value: 285209329703167 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 448888722408026 }
        flops { key: "f32xf32->f32" value: 293632822588364 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 961918767301231 }
        flops { key: "f32xf32->f32" value: 559185925332812 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1185554425781519 }
        flops { key: "f32xf32->f32" value: 652309267722215 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 12861031812955 }
        flops { key: "f32xf32->f32" value: 10251888787045 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 243976783458304 }
        flops { key: "f32xf32->f32" value: 167340734668432 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 668581459526774 }
        flops { key: "f32xf32->f32" value: 416724134866346 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1416487544543843 }
        flops { key: "f32xf32->f32" value: 753341957980475 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 152131173703598 }
        flops { key: "f32xf32->f32" value: 109643809251506 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 376916831592803 }
        flops { key: "f32xf32->f32" value: 259561690699220 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 760844516563330 }
        flops { key: "f32xf32->f32" value: 496800820797547 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 248034609378609 }
        flops { key: "f32xf32->f32" value: 157509435822209 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 780193877565849 }
        flops { key: "f32xf32->f32" value: 502438194484251 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 682011480111155 }
        flops { key: "f32xf32->f32" value: 409453958339291 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 41831923952002 }
        flops { key: "f32xf32->f32" value: 27363451172273 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 74731474387527 }
        flops { key: "f32xf32->f32" value: 52919754756037 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1272347282651360 }
        flops { key: "f32xf32->f32" value: 698908473373744 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 32896501960784 }
        flops { key: "f32xf32->f32" value: 23012040805829 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 428126724082934 }
        flops { key: "f32xf32->f32" value: 298863495650963 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1276246201801467 }
        flops { key: "f32xf32->f32" value: 692610985264770 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 371183760781263 }
        flops { key: "f32xf32->f32" value: 248954747043821 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 266238984378874 }
        flops { key: "f32xf32->f32" value: 185335604384223 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 133384077515527 }
        flops { key: "f32xf32->f32" value: 101210465076821 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 447299239325140 }
        flops { key: "f32xf32->f32" value: 287346443834883 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 789352807737370 }
        flops { key: "f32xf32->f32" value: 511625396348908 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 40227102652480 }
        flops { key: "f32xf32->f32" value: 24738314994009 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 618381296666906 }
        flops { key: "f32xf32->f32" value: 407009457095475 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 903419093103357 }
        flops { key: "f32xf32->f32" value: 606698067733163 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 242242938296672 }
        flops { key: "f32xf32->f32" value: 167106345654034 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 902849367212339 }
        flops { key: "f32xf32->f32" value: 615016438175699 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 697121781528972 }
        flops { key: "f32xf32->f32" value: 429453784221577 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1142696410522465 }
        flops { key: "f32xf32->f32" value: 633335883801518 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 142481664543524 }
        flops { key: "f32xf32->f32" value: 99099383848638 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 884283981058266 }
        flops { key: "f32xf32->f32" value: 525201589190180 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 877777906396893 }
        flops { key: "f32xf32->f32" value: 522645163943901 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 607191248462571 }
        flops { key: "f32xf32->f32" value: 378277901708648 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1148195100016708 }
        flops { key: "f32xf32->f32" value: 589037550024000 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 5990793072665 }
        flops { key: "f32xf32->f32" value: 4529485961123 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 711912364661031 }
        flops { key: "f32xf32->f32" value: 431459872017680 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 272730968757937 }
        flops { key: "f32xf32->f32" value: 182593627072527 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 151103549676329 }
        flops { key: "f32xf32->f32" value: 102593333078540 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 80635462901772 }
        flops { key: "f32xf32->f32" value: 56199195226630 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 147451500137324 }
        flops { key: "f32xf32->f32" value: 104510592174420 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 439967967219832 }
        flops { key: "f32xf32->f32" value: 284002333928453 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 21785055672780 }
        flops { key: "f32xf32->f32" value: 17770121541109 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 425286394296465 }
        flops { key: "f32xf32->f32" value: 297209002560376 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1127177061575304 }
        flops { key: "f32xf32->f32" value: 626225456878326 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 432916772099586 }
        flops { key: "f32xf32->f32" value: 271644253747391 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 952584928416967 }
        flops { key: "f32xf32->f32" value: 539738271567703 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 269445878042659 }
        flops { key: "f32xf32->f32" value: 183828423900017 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 197034925038994 }
        flops { key: "f32xf32->f32" value: 131432991492747 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1045799372028610 }
        flops { key: "f32xf32->f32" value: 611938562895154 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 380658273154302 }
        flops { key: "f32xf32->f32" value: 252704594963520 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 697971446493865 }
        flops { key: "f32xf32->f32" value: 427020013521574 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 958644561352603 }
        flops { key: "f32xf32->f32" value: 553635692823305 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 438485686166411 }
        flops { key: "f32xf32->f32" value: 266024607990089 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 59905256862307 }
        flops { key: "f32xf32->f32" value: 59061706490649 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 133666354288559 }
        flops { key: "f32xf32->f32" value: 100143800037306 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 151103549676329 }
        flops { key: "f32xf32->f32" value: 98112374269005 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43683556712774 }
        flops { key: "f32xf32->f32" value: 32824095866960 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 65012219907968 }
        flops { key: "f32xf32->f32" value: 45957105975004 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1197453766222903 }
        flops { key: "f32xf32->f32" value: 647966854017764 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1143456965889048 }
        flops { key: "f32xf32->f32" value: 608923713257837 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1199460251623263 }
        flops { key: "f32xf32->f32" value: 646503817111031 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 154584195796141 }
        flops { key: "f32xf32->f32" value: 102124959482594 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 938380444832860 }
        flops { key: "f32xf32->f32" value: 535682366748776 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 620660013872832 }
        flops { key: "f32xf32->f32" value: 302899770513769 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 152131173703598 }
        flops { key: "f32xf32->f32" value: 105165702644466 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 81221015431164 }
        flops { key: "f32xf32->f32" value: 55819391973383 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 422649802794725 }
        flops { key: "f32xf32->f32" value: 276381421879021 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1350353246924739 }
        flops { key: "f32xf32->f32" value: 769556557977547 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 376916831592803 }
        flops { key: "f32xf32->f32" value: 251329351980806 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 148779523901898 }
        flops { key: "f32xf32->f32" value: 102436731921389 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 75223611041053 }
        flops { key: "f32xf32->f32" value: 54637788723794 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 271627074120920 }
        flops { key: "f32xf32->f32" value: 180127801375608 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81037118792452 }
        flops { key: "f32xf32->f32" value: 49627557034572 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 134739844898983 }
        flops { key: "f32xf32->f32" value: 88395638758541 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 150426145138694 }
        flops { key: "f32xf32->f32" value: 104683808521010 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 941001762830695 }
        flops { key: "f32xf32->f32" value: 567685595743977 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 42360021461259 }
        flops { key: "f32xf32->f32" value: 28381841404102 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81815134410240 }
        flops { key: "f32xf32->f32" value: 49485750944787 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 948325744314418 }
        flops { key: "f32xf32->f32" value: 539179273263660 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 375861319331408 }
        flops { key: "f32xf32->f32" value: 253435256741606 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 272800260162601 }
        flops { key: "f32xf32->f32" value: 176355723741479 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1185513520615533 }
        flops { key: "f32xf32->f32" value: 656785594341966 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 151465908308647 }
        flops { key: "f32xf32->f32" value: 110524119814719 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 255591960009521 }
        flops { key: "f32xf32->f32" value: 186122694401109 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 956030561157484 }
        flops { key: "f32xf32->f32" value: 486916338860074 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 125657322878876 }
        flops { key: "f32xf32->f32" value: 105652053921086 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 936743139803707 }
        flops { key: "f32xf32->f32" value: 541353999810934 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 272177902154626 }
        flops { key: "f32xf32->f32" value: 182067286816447 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 151787082838563 }
        flops { key: "f32xf32->f32" value: 100887139340411 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 627094071543291 }
        flops { key: "f32xf32->f32" value: 406335600378429 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 376388335465778 }
        flops { key: "f32xf32->f32" value: 245314558830249 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 254140076686390 }
        flops { key: "f32xf32->f32" value: 162282449028942 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 383410756650598 }
        flops { key: "f32xf32->f32" value: 263107528546924 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 40814270336019 }
        flops { key: "f32xf32->f32" value: 29530853245324 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1140269417847542 }
        flops { key: "f32xf32->f32" value: 627620983596975 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1148482940352636 }
        flops { key: "f32xf32->f32" value: 730327935213722 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 255106159182703 }
        flops { key: "f32xf32->f32" value: 188739993671998 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81024888620585 }
        flops { key: "f32xf32->f32" value: 58757897778264 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 927839122056599 }
        flops { key: "f32xf32->f32" value: 544752804134825 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81616131346913 }
        flops { key: "f32xf32->f32" value: 49774792508807 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 712857642489626 }
        flops { key: "f32xf32->f32" value: 430422137194969 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 461130265836375 }
        flops { key: "f32xf32->f32" value: 292353638009665 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 242680941123290 }
        flops { key: "f32xf32->f32" value: 166484506395844 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 617669849140720 }
        flops { key: "f32xf32->f32" value: 389248440819285 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 258048984378755 }
        flops { key: "f32xf32->f32" value: 178214410622406 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1278192748470137 }
        flops { key: "f32xf32->f32" value: 692492333712903 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1153437120010742 }
        flops { key: "f32xf32->f32" value: 733524152854276 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 960198367091437 }
        flops { key: "f32xf32->f32" value: 565946408749505 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1336642743639617 }
        flops { key: "f32xf32->f32" value: 657395049754623 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 81628540672038 }
        flops { key: "f32xf32->f32" value: 48691357881371 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 616340287866829 }
        flops { key: "f32xf32->f32" value: 428105387091951 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 674407991834812 }
        flops { key: "f32xf32->f32" value: 420662810577864 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 79879617914000 }
        flops { key: "f32xf32->f32" value: 56299382550335 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 673456259662877 }
        flops { key: "f32xf32->f32" value: 429410847430513 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 244922861313868 }
        flops { key: "f32xf32->f32" value: 164457317200183 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 255591960009521 }
        flops { key: "f32xf32->f32" value: 179151050971886 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 629299237509157 }
        flops { key: "f32xf32->f32" value: 416744352416068 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 52331700165708 }
        flops { key: "f32xf32->f32" value: 34234849636525 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 252704594963520 }
        flops { key: "f32xf32->f32" value: 162257925802795 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 151146090090090 }
        flops { key: "f32xf32->f32" value: 109109015750431 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 252704594963520 }
        flops { key: "f32xf32->f32" value: 163244671075636 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 42683328987120 }
        flops { key: "f32xf32->f32" value: 32009951824469 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 149775676384432 }
        flops { key: "f32xf32->f32" value: 108920858592006 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1043924724068785 }
        flops { key: "f32xf32->f32" value: 641195408737193 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1142639409653979 }
        flops { key: "f32xf32->f32" value: 733320635321737 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 81221015431164 }
        flops { key: "f32xf32->f32" value: 55634291398963 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 260080373985709 }
        flops { key: "f32xf32->f32" value: 181574672190749 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 65265124240213 }
        flops { key: "f32xf32->f32" value: 48128275392200 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 961918767301231 }
        flops { key: "f32xf32->f32" value: 559714249820811 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 270566164545798 }
        flops { key: "f32xf32->f32" value: 182330077092885 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 82430663595885 }
        flops { key: "f32xf32->f32" value: 58964405491488 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 150785258250245 }
        flops { key: "f32xf32->f32" value: 103056130530761 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 956775962575183 }
        flops { key: "f32xf32->f32" value: 556811732157905 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 621243551891227 }
        flops { key: "f32xf32->f32" value: 415113062001643 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 82228658600091 }
        flops { key: "f32xf32->f32" value: 58648777802053 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43226321417069 }
        flops { key: "f32xf32->f32" value: 31584357689139 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 380152885112409 }
        flops { key: "f32xf32->f32" value: 249649342943501 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 614137026667619 }
        flops { key: "f32xf32->f32" value: 392091226583896 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 82241254901960 }
        flops { key: "f32xf32->f32" value: 49701065728568 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 497909494087642 }
        flops { key: "f32xf32->f32" value: 344065312505006 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 426680637393204 }
        flops { key: "f32xf32->f32" value: 264435863563600 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 253659774155445 }
        flops { key: "f32xf32->f32" value: 172253440924039 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 150468304932735 }
        flops { key: "f32xf32->f32" value: 103234479761561 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 43457253683017 }
        flops { key: "f32xf32->f32" value: 29582924399382 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 638229778735418 }
        flops { key: "f32xf32->f32" value: 400631248169395 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 150447222082107 }
        flops { key: "f32xf32->f32" value: 105330765548361 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1184900281674598 }
        flops { key: "f32xf32->f32" value: 657804081020025 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 149754787168758 }
        flops { key: "f32xf32->f32" value: 104673603431468 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 150764086492558 }
        flops { key: "f32xf32->f32" value: 110172565565360 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 379146124293785 }
        flops { key: "f32xf32->f32" value: 255348828537455 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 923897240333423 }
        flops { key: "f32xf32->f32" value: 548842539901603 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1268917141886402 }
        flops { key: "f32xf32->f32" value: 688944686864636 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 691676833239391 }
        flops { key: "f32xf32->f32" value: 429067661938061 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 133119492189437 }
        flops { key: "f32xf32->f32" value: 100443575678203 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 685549448683160 }
        flops { key: "f32xf32->f32" value: 411060659041967 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 152152731188890 }
        flops { key: "f32xf32->f32" value: 99845808443369 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 437102309790352 }
        flops { key: "f32xf32->f32" value: 273008345792016 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 962781281327056 }
        flops { key: "f32xf32->f32" value: 473744462386940 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 935264259567750 }
        flops { key: "f32xf32->f32" value: 546624747335263 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 674355047260166 }
        flops { key: "f32xf32->f32" value: 403056240240240 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1176461630076011 }
        flops { key: "f32xf32->f32" value: 636538994201448 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 44143308008551 }
        flops { key: "f32xf32->f32" value: 32506109953984 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 609258429108447 }
        flops { key: "f32xf32->f32" value: 397075513890814 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 885742894617446 }
        flops { key: "f32xf32->f32" value: 526231175421937 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 432176222177500 }
        flops { key: "f32xf32->f32" value: 270310736736106 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1342964173070158 }
        flops { key: "f32xf32->f32" value: 700832977094254 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 128039807297877 }
        flops { key: "f32xf32->f32" value: 88170621120052 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81024888620585 }
        flops { key: "f32xf32->f32" value: 48908710212262 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 240076427948574 }
        flops { key: "f32xf32->f32" value: 167353775561097 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 689899172114689 }
        flops { key: "f32xf32->f32" value: 419348495996875 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 11779684746357 }
        flops { key: "f32xf32->f32" value: 8254472816728 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 273321070128547 }
        flops { key: "f32xf32->f32" value: 185351600897635 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 934297867304764 }
        flops { key: "f32xf32->f32" value: 545202284408619 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 80647575784888 }
        flops { key: "f32xf32->f32" value: 56477057858194 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 196782154128104 }
        flops { key: "f32xf32->f32" value: 133525066716408 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 38926255220417 }
        flops { key: "f32xf32->f32" value: 33412429176001 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 245763750057221 }
        flops { key: "f32xf32->f32" value: 160319794550205 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 972427077828720 }
        flops { key: "f32xf32->f32" value: 558931228942317 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 150447222082107 }
        flops { key: "f32xf32->f32" value: 106829352701223 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 431481544705646 }
        flops { key: "f32xf32->f32" value: 264403305589756 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1193626706315570 }
        flops { key: "f32xf32->f32" value: 660827740513510 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1037713701428528 }
        flops { key: "f32xf32->f32" value: 612440303869668 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1344409209351462 }
        flops { key: "f32xf32->f32" value: 695865248354497 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 17508182624576 }
        flops { key: "f32xf32->f32" value: 16412047933480 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 143663610382659 }
        flops { key: "f32xf32->f32" value: 104358229565555 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 676000203982057 }
        flops { key: "f32xf32->f32" value: 402961701552751 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 134184181954511 }
        flops { key: "f32xf32->f32" value: 88629122905489 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 266238984378874 }
        flops { key: "f32xf32->f32" value: 184333360343347 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 888675211255948 }
        flops { key: "f32xf32->f32" value: 533287884029178 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 133649716704007 }
        flops { key: "f32xf32->f32" value: 89099811136005 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 760305770224818 }
        flops { key: "f32xf32->f32" value: 499298685887003 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 253689739870053 }
        flops { key: "f32xf32->f32" value: 163033984816276 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 375861319331408 }
        flops { key: "f32xf32->f32" value: 249185849152935 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 259045072135102 }
        flops { key: "f32xf32->f32" value: 176109861243234 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 249417380720092 }
        flops { key: "f32xf32->f32" value: 163455902572689 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 128299895327996 }
        flops { key: "f32xf32->f32" value: 102280608115831 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 146826449336797 }
        flops { key: "f32xf32->f32" value: 101353768548234 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 20357610799332 }
        flops { key: "f32xf32->f32" value: 14924688980318 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 458815008652921 }
        flops { key: "f32xf32->f32" value: 287365669476783 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 443603315017558 }
        flops { key: "f32xf32->f32" value: 289203911925122 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 961111562741258 }
        flops { key: "f32xf32->f32" value: 525233702772937 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 692624946944041 }
        flops { key: "f32xf32->f32" value: 424991816346724 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 257060527651424 }
        flops { key: "f32xf32->f32" value: 176341242240105 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 465983215362916 }
        flops { key: "f32xf32->f32" value: 294923250429169 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 611339733257419 }
        flops { key: "f32xf32->f32" value: 388684823167420 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 607191248462571 }
        flops { key: "f32xf32->f32" value: 394685471053115 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1411468820638164 }
        flops { key: "f32xf32->f32" value: 751048949004349 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 960413080500894 }
        flops { key: "f32xf32->f32" value: 533917679833421 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 74710675201781 }
        flops { key: "f32xf32->f32" value: 54109142511590 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1131296535229817 }
        flops { key: "f32xf32->f32" value: 608923713257837 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1156192824819974 }
        flops { key: "f32xf32->f32" value: 624665727988364 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 68182742189484 }
        flops { key: "f32xf32->f32" value: 45769046206308 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 940023483475596 }
        flops { key: "f32xf32->f32" value: 543873280486260 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1147696518404703 }
        flops { key: "f32xf32->f32" value: 618938256439817 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 149775676384432 }
        flops { key: "f32xf32->f32" value: 102436731921389 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 258079996154308 }
        flops { key: "f32xf32->f32" value: 186624111236638 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 150426145138694 }
        flops { key: "f32xf32->f32" value: 100896619432437 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 267332708577119 }
        flops { key: "f32xf32->f32" value: 188723406977766 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 425328510200039 }
        flops { key: "f32xf32->f32" value: 275036327868852 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1038246762796881 }
        flops { key: "f32xf32->f32" value: 607212709292051 }
      }
    }
  }
)pb";

#endif  // XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_DATA_H_
