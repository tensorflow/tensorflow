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
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 340681153010232 }
        flops { key: "F32xF32->F32" value: 150384008963585 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 82850449382716 }
        flops { key: "F32xF32->F32" value: 51861564142194 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 448139325542570 }
        flops { key: "F32xF32->F32" value: 231909681209503 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 139810133333333 }
        flops { key: "F32xF32->F32" value: 81740394640682 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 764093096602028 }
        flops { key: "F32xF32->F32" value: 384750272865717 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 170111188846641 }
        flops { key: "F32xF32->F32" value: 139230008298755 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 148470938053097 }
        flops { key: "F32xF32->F32" value: 75915004524886 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 542020102978293 }
        flops { key: "F32xF32->F32" value: 273425470842882 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 362750616216216 }
        flops { key: "F32xF32->F32" value: 145493472086720 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 653127630170316 }
        flops { key: "F32xF32->F32" value: 278315662001036 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 608524694814395 }
        flops { key: "F32xF32->F32" value: 299927883798882 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 216131607085346 }
        flops { key: "F32xF32->F32" value: 106353191759112 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 621648182949775 }
        flops { key: "F32xF32->F32" value: 316341407969359 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 41838443890274 }
        flops { key: "F32xF32->F32" value: 27776847682119 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 77852510440835 }
        flops { key: "F32xF32->F32" value: 50533783132530 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 115307326460481 }
        flops { key: "F32xF32->F32" value: 55924053333333 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 37617076233183 }
        flops { key: "F32xF32->F32" value: 23269370319001 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 92182505494505 }
        flops { key: "F32xF32->F32" value: 103723128284389 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 334707551122194 }
        flops { key: "F32xF32->F32" value: 161319384615384 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 214748364800000 }
        flops { key: "F32xF32->F32" value: 110014531147540 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 233828794425087 }
        flops { key: "F32xF32->F32" value: 111476518272425 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 734182443760683 }
        flops { key: "F32xF32->F32" value: 253481998421258 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 231809547495682 }
        flops { key: "F32xF32->F32" value: 95189878014184 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 426765430842607 }
        flops { key: "F32xF32->F32" value: 179555488963210 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 219310013071895 }
        flops { key: "F32xF32->F32" value: 110649404781533 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 489845722627737 }
        flops { key: "F32xF32->F32" value: 198331476807277 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 456911414468085 }
        flops { key: "F32xF32->F32" value: 176718535878867 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 598852104852203 }
        flops { key: "F32xF32->F32" value: 261251052068126 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 299927883798882 }
        flops { key: "F32xF32->F32" value: 150721760808534 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 288330242749731 }
        flops { key: "F32xF32->F32" value: 119944350312779 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 17848102127659 }
        flops { key: "F32xF32->F32" value: 22250949602122 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 673192366144200 }
        flops { key: "F32xF32->F32" value: 279656680296913 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 659141696746470 }
        flops { key: "F32xF32->F32" value: 315065089201877 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 138368791752577 }
        flops { key: "F32xF32->F32" value: 62718564485981 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 681092181414525 }
        flops { key: "F32xF32->F32" value: 310285168039300 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 214748364800000 }
        flops { key: "F32xF32->F32" value: 106691357710651 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 505052598306679 }
        flops { key: "F32xF32->F32" value: 243037986419194 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 378611362482369 }
        flops { key: "F32xF32->F32" value: 136817255861365 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 246723764705882 }
        flops { key: "F32xF32->F32" value: 109834474631751 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 390735743813682 }
        flops { key: "F32xF32->F32" value: 199431988112927 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 80854053012048 }
        flops { key: "F32xF32->F32" value: 51861564142194 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 229040491467576 }
        flops { key: "F32xF32->F32" value: 107805404016064 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 23967451428571 }
        flops { key: "F32xF32->F32" value: 17734900634249 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 48210390804597 }
        flops { key: "F32xF32->F32" value: 32263876923076 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 341955994904458 }
        flops { key: "F32xF32->F32" value: 185127900689655 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 206806976887519 }
        flops { key: "F32xF32->F32" value: 109476123980424 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 138368791752577 }
        flops { key: "F32xF32->F32" value: 77136625287356 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 455361248515691 }
        flops { key: "F32xF32->F32" value: 226193769538656 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 77852510440835 }
        flops { key: "F32xF32->F32" value: 51150048780487 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 742303369512616 }
        flops { key: "F32xF32->F32" value: 299886000279290 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 141879205073995 }
        flops { key: "F32xF32->F32" value: 68269444557477 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 240533562724014 }
        flops { key: "F32xF32->F32" value: 109744667211774 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 604924971267605 }
        flops { key: "F32xF32->F32" value: 273216749109414 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 321865055155875 }
        flops { key: "F32xF32->F32" value: 105600100708103 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 717502054126294 }
        flops { key: "F32xF32->F32" value: 318428773428232 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 238821580071174 }
        flops { key: "F32xF32->F32" value: 111754977518734 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 489399190519598 }
        flops { key: "F32xF32->F32" value: 241616072007200 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 307134389016018 }
        flops { key: "F32xF32->F32" value: 180642971736204 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 590617064906490 }
        flops { key: "F32xF32->F32" value: 231609539257981 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 597519100723427 }
        flops { key: "F32xF32->F32" value: 258920140824692 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 455747803056027 }
        flops { key: "F32xF32->F32" value: 170003455351488 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 41838443890274 }
        flops { key: "F32xF32->F32" value: 26672839427662 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 600526747203579 }
        flops { key: "F32xF32->F32" value: 207767380804953 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 656321408312958 }
        flops { key: "F32xF32->F32" value: 180977890443283 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 726360104177236 }
        flops { key: "F32xF32->F32" value: 326366846043151 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 12520310447761 }
        flops { key: "F32xF32->F32" value: 10058282973621 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 373865537604456 }
        flops { key: "F32xF32->F32" value: 135641968671045 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 492090661778185 }
        flops { key: "F32xF32->F32" value: 249939903165735 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 632729418974661 }
        flops { key: "F32xF32->F32" value: 234595111208215 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 526344031372549 }
        flops { key: "F32xF32->F32" value: 240318223813786 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 239247286987522 }
        flops { key: "F32xF32->F32" value: 124045959334565 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 139230008298755 }
        flops { key: "F32xF32->F32" value: 68200065040650 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 120916872072072 }
        flops { key: "F32xF32->F32" value: 55461871074380 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 764165319129298 }
        flops { key: "F32xF32->F32" value: 292095785092035 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 232613046793760 }
        flops { key: "F32xF32->F32" value: 96144504297994 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 364227213025780 }
        flops { key: "F32xF32->F32" value: 144636042970197 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 143702064239828 }
        flops { key: "F32xF32->F32" value: 81442796116504 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 221116520593080 }
        flops { key: "F32xF32->F32" value: 93990005602240 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 735943676490747 }
        flops { key: "F32xF32->F32" value: 250903569108540 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 729568081535586 }
        flops { key: "F32xF32->F32" value: 324957804040251 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 370767204419889 }
        flops { key: "F32xF32->F32" value: 140321722948248 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 782183080677472 }
        flops { key: "F32xF32->F32" value: 318630309224909 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 139810133333333 }
        flops { key: "F32xF32->F32" value: 75658245772266 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 716305419613075 }
        flops { key: "F32xF32->F32" value: 286448119382081 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 239247286987522 }
        flops { key: "F32xF32->F32" value: 126620498113207 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 25497288753799 }
        flops { key: "F32xF32->F32" value: 17623126050420 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 135573462626262 }
        flops { key: "F32xF32->F32" value: 80466263788968 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 22795130434782 }
        flops { key: "F32xF32->F32" value: 15857482041587 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 695203511816121 }
        flops { key: "F32xF32->F32" value: 307024611909357 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 360800344086021 }
        flops { key: "F32xF32->F32" value: 149462948775055 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 138654677685950 }
        flops { key: "F32xF32->F32" value: 81541754556500 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 82241254901960 }
        flops { key: "F32xF32->F32" value: 54560052032520 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 342829445721583 }
        flops { key: "F32xF32->F32" value: 177068242744063 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 299927883798882 }
        flops { key: "F32xF32->F32" value: 163980119731215 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 212369822784810 }
        flops { key: "F32xF32->F32" value: 111015490488006 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 393600375366568 }
        flops { key: "F32xF32->F32" value: 198107347601476 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 137518163934426 }
        flops { key: "F32xF32->F32" value: 61908546125461 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 394467973548861 }
        flops { key: "F32xF32->F32" value: 210125601565557 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 449640629815745 }
        flops { key: "F32xF32->F32" value: 191397829590017 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 652730592097264 }
        flops { key: "F32xF32->F32" value: 189807640799010 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 61342654478976 }
        flops { key: "F32xF32->F32" value: 37158839424141 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 612516727895037 }
        flops { key: "F32xF32->F32" value: 178392062468848 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 137518163934426 }
        flops { key: "F32xF32->F32" value: 80369897005988 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 337654661635220 }
        flops { key: "F32xF32->F32" value: 177419336417713 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 121794671506352 }
        flops { key: "F32xF32->F32" value: 55142862777321 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 544217853015712 }
        flops { key: "F32xF32->F32" value: 283908467477525 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 82646384236453 }
        flops { key: "F32xF32->F32" value: 54295197411003 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 374909854748603 }
        flops { key: "F32xF32->F32" value: 183985919122686 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 398863976225854 }
        flops { key: "F32xF32->F32" value: 228261442176870 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 92056054869684 }
        flops { key: "F32xF32->F32" value: 104857600000000 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 488508564149226 }
        flops { key: "F32xF32->F32" value: 213467559443339 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 527378106090373 }
        flops { key: "F32xF32->F32" value: 260111875968992 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 42366707070707 }
        flops { key: "F32xF32->F32" value: 26929720706260 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 786048187408492 }
        flops { key: "F32xF32->F32" value: 403834320513611 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 39662449172576 }
        flops { key: "F32xF32->F32" value: 22399487316421 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 80854053012048 }
        flops { key: "F32xF32->F32" value: 44679669773635 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 463219078515962 }
        flops { key: "F32xF32->F32" value: 181008399190829 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 341520936386768 }
        flops { key: "F32xF32->F32" value: 179555488963210 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 717022920868113 }
        flops { key: "F32xF32->F32" value: 281748051430070 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 235469698245614 }
        flops { key: "F32xF32->F32" value: 110832145334434 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 776881898084347 }
        flops { key: "F32xF32->F32" value: 309927621679764 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 522247968871595 }
        flops { key: "F32xF32->F32" value: 172738388674388 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 546433498218829 }
        flops { key: "F32xF32->F32" value: 281600268554943 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 81840078048780 }
        flops { key: "F32xF32->F32" value: 51701744221879 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 746950834086956 }
        flops { key: "F32xF32->F32" value: 310735546011548 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 86703958656330 }
        flops { key: "F32xF32->F32" value: 65664250489236 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 734182443760683 }
        flops { key: "F32xF32->F32" value: 239193990643795 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 300263373601789 }
        flops { key: "F32xF32->F32" value: 142784817021276 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 378611362482369 }
        flops { key: "F32xF32->F32" value: 190514872959545 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 465629585429314 }
        flops { key: "F32xF32->F32" value: 180037193829644 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 208736746500777 }
        flops { key: "F32xF32->F32" value: 104939584050039 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 435771844155844 }
        flops { key: "F32xF32->F32" value: 177126661827779 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 62484975791433 }
        flops { key: "F32xF32->F32" value: 67378377510040 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 391020329206118 }
        flops { key: "F32xF32->F32" value: 206806976887519 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 651937962355798 }
        flops { key: "F32xF32->F32" value: 181990139661016 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 24745156342182 }
        flops { key: "F32xF32->F32" value: 17848102127659 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 46218225895316 }
        flops { key: "F32xF32->F32" value: 31956601904761 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 210372614420062 }
        flops { key: "F32xF32->F32" value: 111384006639004 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 490293070319634 }
        flops { key: "F32xF32->F32" value: 201528120120120 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 721843243025210 }
        flops { key: "F32xF32->F32" value: 300686424097102 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 633102490566037 }
        flops { key: "F32xF32->F32" value: 279111469716662 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 318806954869358 }
        flops { key: "F32xF32->F32" value: 118829329791943 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 47662545454545 }
        flops { key: "F32xF32->F32" value: 32017587786259 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 720148775318578 }
        flops { key: "F32xF32->F32" value: 289301313215681 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 458080982935153 }
        flops { key: "F32xF32->F32" value: 180400172043010 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 448139325542570 }
        flops { key: "F32xF32->F32" value: 191261457784111 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 728701611129962 }
        flops { key: "F32xF32->F32" value: 311590778874056 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 651542368932038 }
        flops { key: "F32xF32->F32" value: 180190146985095 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 353669902503293 }
        flops { key: "F32xF32->F32" value: 148635357696567 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 142481664543524 }
        flops { key: "F32xF32->F32" value: 62894905342080 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 732554544772300 }
        flops { key: "F32xF32->F32" value: 305562492434247 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 763755531195905 }
        flops { key: "F32xF32->F32" value: 340594194371132 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 488953471766848 }
        flops { key: "F32xF32->F32" value: 211200201416207 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 722450344154751 }
        flops { key: "F32xF32->F32" value: 320928592545545 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 729072703445934 }
        flops { key: "F32xF32->F32" value: 267267205986333 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 227873901528013 }
        flops { key: "F32xF32->F32" value: 85163532994923 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 249475330855018 }
        flops { key: "F32xF32->F32" value: 108678322267206 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 115109543739279 }
        flops { key: "F32xF32->F32" value: 56394003361344 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 380220192634560 }
        flops { key: "F32xF32->F32" value: 186672778859527 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 755491169041336 }
        flops { key: "F32xF32->F32" value: 337894568106777 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 516718875842155 }
        flops { key: "F32xF32->F32" value: 154673267646211 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 319946908224076 }
        flops { key: "F32xF32->F32" value: 97969144525547 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 318428773428232 }
        flops { key: "F32xF32->F32" value: 118357784832451 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 81640953771289 }
        flops { key: "F32xF32->F32" value: 46474282548476 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 143702064239828 }
        flops { key: "F32xF32->F32" value: 76000978482446 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 597186776418242 }
        flops { key: "F32xF32->F32" value: 267033529967669 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 680013821405953 }
        flops { key: "F32xF32->F32" value: 313135556722076 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 444429562913907 }
        flops { key: "F32xF32->F32" value: 231709500215796 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 584826701525054 }
        flops { key: "F32xF32->F32" value: 155592207506158 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 82646384236453 }
        flops { key: "F32xF32->F32" value: 50533783132530 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 630870636897767 }
        flops { key: "F32xF32->F32" value: 335754166353971 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 222953036544850 }
        flops { key: "F32xF32->F32" value: 78215459207459 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 508400484848484 }
        flops { key: "F32xF32->F32" value: 165598677359654 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 41221660933660 }
        flops { key: "F32xF32->F32" value: 28532680272108 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 694192224987877 }
        flops { key: "F32xF32->F32" value: 296388606445379 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 340654131979695 }
        flops { key: "F32xF32->F32" value: 183482881749829 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 120051635062611 }
        flops { key: "F32xF32->F32" value: 56871918644067 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 544217853015712 }
        flops { key: "F32xF32->F32" value: 267966514599450 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 374909854748603 }
        flops { key: "F32xF32->F32" value: 136817255861365 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 308901560414269 }
        flops { key: "F32xF32->F32" value: 131457128305582 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 147168561403508 }
        flops { key: "F32xF32->F32" value: 80177854241338 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 23899168091168 }
        flops { key: "F32xF32->F32" value: 17772474576271 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 243589343012704 }
        flops { key: "F32xF32->F32" value: 110740699669967 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 624995240977881 }
        flops { key: "F32xF32->F32" value: 208049181166440 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 810450034773430 }
        flops { key: "F32xF32->F32" value: 337628947952568 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 593227527071823 }
        flops { key: "F32xF32->F32" value: 168933578351164 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 633102490566037 }
        flops { key: "F32xF32->F32" value: 336279932352020 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 46218225895316 }
        flops { key: "F32xF32->F32" value: 32640498054474 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 82040176039119 }
        flops { key: "F32xF32->F32" value: 51463852760736 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 448139325542570 }
        flops { key: "F32xF32->F32" value: 232915796963123 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 151146090090090 }
        flops { key: "F32xF32->F32" value: 76695844571428 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 81442796116504 }
        flops { key: "F32xF32->F32" value: 46798370990237 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 136956865306122 }
        flops { key: "F32xF32->F32" value: 76000978482446 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 375960022408963 }
        flops { key: "F32xF32->F32" value: 144631172413793 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 722571886944818 }
        flops { key: "F32xF32->F32" value: 313776102863822 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 342392163265306 }
        flops { key: "F32xF32->F32" value: 179796018754186 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 524800500488758 }
        flops { key: "F32xF32->F32" value: 238080227050997 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 184618607977991 }
        flops { key: "F32xF32->F32" value: 72393596548004 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 112222180602006 }
        flops { key: "F32xF32->F32" value: 54295197411003 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 73423264770240 }
        flops { key: "F32xF32->F32" value: 38086755959137 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 449640629815745 }
        flops { key: "F32xF32->F32" value: 226146129738837 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 238397385435168 }
        flops { key: "F32xF32->F32" value: 111569183707398 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 447392426666666 }
        flops { key: "F32xF32->F32" value: 224632180753138 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 464421204152249 }
        flops { key: "F32xF32->F32" value: 230367265393692 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 60349697841726 }
        flops { key: "F32xF32->F32" value: 67108864000000 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 631984593290170 }
        flops { key: "F32xF32->F32" value: 232562664933939 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 372827022222222 }
        flops { key: "F32xF32->F32" value: 186284147120055 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 76783597254004 }
        flops { key: "F32xF32->F32" value: 50156101644245 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 345032719794344 }
        flops { key: "F32xF32->F32" value: 118671731211317 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 336807347553324 }
        flops { key: "F32xF32->F32" value: 147979854465270 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 345032719794344 }
        flops { key: "F32xF32->F32" value: 145493472086720 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 595530684414864 }
        flops { key: "F32xF32->F32" value: 258173076220245 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 235057316987740 }
        flops { key: "F32xF32->F32" value: 110832145334434 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 303316899435028 }
        flops { key: "F32xF32->F32" value: 163480789281364 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 48489063583815 }
        flops { key: "F32xF32->F32" value: 32140260536398 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 352277501312335 }
        flops { key: "F32xF32->F32" value: 147898322865013 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 81245598062954 }
        flops { key: "F32xF32->F32" value: 51306470948012 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 83261617866004 }
        flops { key: "F32xF32->F32" value: 52103155279503 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 237553500884955 }
        flops { key: "F32xF32->F32" value: 111199443247721 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 205225883792048 }
        flops { key: "F32xF32->F32" value: 96006958512160 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 235057316987740 }
        flops { key: "F32xF32->F32" value: 110923742148760 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 374909854748603 }
        flops { key: "F32xF32->F32" value: 145572373101952 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 524800500488758 }
        flops { key: "F32xF32->F32" value: 256507841376015 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 133949828343313 }
        flops { key: "F32xF32->F32" value: 80369897005988 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 299259148272017 }
        flops { key: "F32xF32->F32" value: 159593017835909 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 218240208130081 }
        flops { key: "F32xF32->F32" value: 110923742148760 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 40233131894484 }
        flops { key: "F32xF32->F32" value: 39016781395348 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 610080581818181 }
        flops { key: "F32xF32->F32" value: 210971966597897 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 301612871910112 }
        flops { key: "F32xF32->F32" value: 160932527577937 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 145888834782608 }
        flops { key: "F32xF32->F32" value: 76260072727272 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 215092512820512 }
        flops { key: "F32xF32->F32" value: 114033753610875 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 48349325648414 }
        flops { key: "F32xF32->F32" value: 32768000000000 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 135847902834008 }
        flops { key: "F32xF32->F32" value: 80177854241338 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 636857546856465 }
        flops { key: "F32xF32->F32" value: 222214781456953 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 596854821567537 }
        flops { key: "F32xF32->F32" value: 270668470884799 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 490293070319634 }
        flops { key: "F32xF32->F32" value: 237869256535223 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 47259763380281 }
        flops { key: "F32xF32->F32" value: 32961131630648 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 398272189910979 }
        flops { key: "F32xF32->F32" value: 114569123346137 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 6186289085545 }
        flops { key: "F32xF32->F32" value: 5729923497267 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 760440385269121 }
        flops { key: "F32xF32->F32" value: 384165232200357 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 347714321243523 }
        flops { key: "F32xF32->F32" value: 178362429235880 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 230219087478559 }
        flops { key: "F32xF32->F32" value: 110558260296540 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 660967574022776 }
        flops { key: "F32xF32->F32" value: 231235452568105 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 221847484297520 }
        flops { key: "F32xF32->F32" value: 78169905649388 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 139230008298755 }
        flops { key: "F32xF32->F32" value: 79231244391971 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 722875922915088 }
        flops { key: "F32xF32->F32" value: 322699372328036 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 658737315337423 }
        flops { key: "F32xF32->F32" value: 279401983866770 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 83676887780548 }
        flops { key: "F32xF32->F32" value: 55007265573770 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 524288000000000 }
        flops { key: "F32xF32->F32" value: 168668209864907 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 139519467775467 }
        flops { key: "F32xF32->F32" value: 63974131553860 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 245370617915904 }
        flops { key: "F32xF32->F32" value: 110740699669967 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 742046872149274 }
        flops { key: "F32xF32->F32" value: 300473436127046 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 22075284210526 }
        flops { key: "F32xF32->F32" value: 16288559223300 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 689622237636480 }
        flops { key: "F32xF32->F32" value: 336518712561273 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 371794260387811 }
        flops { key: "F32xF32->F32" value: 112222180602006 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 483232144014401 }
        flops { key: "F32xF32->F32" value: 159687957168352 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 543391611336032 }
        flops { key: "F32xF32->F32" value: 235160276828734 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 330178912669126 }
        flops { key: "F32xF32->F32" value: 119198692717584 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 73746004395604 }
        flops { key: "F32xF32->F32" value: 38926255220417 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 631984593290170 }
        flops { key: "F32xF32->F32" value: 228212927523910 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 809495348323011 }
        flops { key: "F32xF32->F32" value: 336544257405425 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 187193483960948 }
        flops { key: "F32xF32->F32" value: 133417224652087 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 781755969421186 }
        flops { key: "F32xF32->F32" value: 329521857535376 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 142179796610169 }
        flops { key: "F32xF32->F32" value: 63191020715630 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 734559140755943 }
        flops { key: "F32xF32->F32" value: 275919780033406 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 305040290909090 }
        flops { key: "F32xF32->F32" value: 120753691408007 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 10979853403141 }
        flops { key: "F32xF32->F32" value: 7570945848375 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 674884867379006 }
        flops { key: "F32xF32->F32" value: 305300490190503 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 223696213333333 }
        flops { key: "F32xF32->F32" value: 82748290998766 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 81640953771289 }
        flops { key: "F32xF32->F32" value: 52022375193798 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 748773935843793 }
        flops { key: "F32xF32->F32" value: 308635189422247 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 778355798477709 }
        flops { key: "F32xF32->F32" value: 329167503256014 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 67923951417004 }
        flops { key: "F32xF32->F32" value: 40041088305489 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 136123456389452 }
        flops { key: "F32xF32->F32" value: 79796508917954 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 83886080000000 }
        flops { key: "F32xF32->F32" value: 53092455696202 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 488064465454545 }
        flops { key: "F32xF32->F32" value: 199283931700074 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 142784817021276 }
        flops { key: "F32xF32->F32" value: 79796508917954 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 491640029304029 }
        flops { key: "F32xF32->F32" value: 258857720347155 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 481067125448028 }
        flops { key: "F32xF32->F32" value: 181375308108108 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 12520310447761 }
        flops { key: "F32xF32->F32" value: 9597949656750 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 393024093704246 }
        flops { key: "F32xF32->F32" value: 184809264027538 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 137800542094455 }
        flops { key: "F32xF32->F32" value: 80177854241338 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 523265996101364 }
        flops { key: "F32xF32->F32" value: 177360724149322 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 590942115575123 }
        flops { key: "F32xF32->F32" value: 297682790130302 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 148143187637969 }
        flops { key: "F32xF32->F32" value: 80177854241338 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 72471775377969 }
        flops { key: "F32xF32->F32" value: 39429414806110 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 81049352657004 }
        flops { key: "F32xF32->F32" value: 50840048484848 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 343707370038412 }
        flops { key: "F32xF32->F32" value: 185255663216011 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 512
        flops { key: "BF16xBF16->BF16" value: 217180789644012 }
        flops { key: "F32xF32->F32" value: 111848106666666 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 473014019383259 }
        flops { key: "F32xF32->F32" value: 165191049846153 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 309971658198614 }
        flops { key: "F32xF32->F32" value: 185383602209944 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 233422135652173 }
        flops { key: "F32xF32->F32" value: 96006958512160 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 60567566787003 }
        flops { key: "F32xF32->F32" value: 67786731313131 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 739873780534022 }
        flops { key: "F32xF32->F32" value: 326613482585551 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 655520039072039 }
        flops { key: "F32xF32->F32" value: 316271524005891 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 181620741542625 }
        flops { key: "F32xF32->F32" value: 140542123560209 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 608352308215297 }
        flops { key: "F32xF32->F32" value: 293452261273572 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 751393858642407 }
        flops { key: "F32xF32->F32" value: 298240906603708 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 236298816901408 }
        flops { key: "F32xF32->F32" value: 111199443247721 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 543942160081053 }
        flops { key: "F32xF32->F32" value: 276808926011858 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 42581766497461 }
        flops { key: "F32xF32->F32" value: 27548794745484 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 355073354497354 }
        flops { key: "F32xF32->F32" value: 163680156097560 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 78581807962529 }
        flops { key: "F32xF32->F32" value: 46798370990237 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 724522148448043 }
        flops { key: "F32xF32->F32" value: 299865062905815 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 132104062992125 }
        flops { key: "F32xF32->F32" value: 68061728194726 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 512
        flops { key: "BF16xBF16->BF16" value: 443328581337737 }
        flops { key: "F32xF32->F32" value: 231509664510564 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 142179796610169 }
        flops { key: "F32xF32->F32" value: 68130826395939 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 713924085106383 }
        flops { key: "F32xF32->F32" value: 297270715393134 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 627919195321637 }
        flops { key: "F32xF32->F32" value: 221344428777571 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 651937962355798 }
        flops { key: "F32xF32->F32" value: 185736347344750 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 82241254901960 }
        flops { key: "F32xF32->F32" value: 54738061990212 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 31126560296846 }
        flops { key: "F32xF32->F32" value: 38391798627002 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 47393265536723 }
        flops { key: "F32xF32->F32" value: 32201950095969 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 526344031372549 }
        flops { key: "F32xF32->F32" value: 167301624181988 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 159403477434679 }
        flops { key: "F32xF32->F32" value: 103723128284389 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 140689442348008 }
        flops { key: "F32xF32->F32" value: 82040176039119 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 256
        flops { key: "BF16xBF16->BF16" value: 138654677685950 }
        flops { key: "F32xF32->F32" value: 81049352657004 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 211366500787401 }
        flops { key: "F32xF32->F32" value: 112410157453936 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 333046471464019 }
        flops { key: "F32xF32->F32" value: 138155149768399 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 368730021978022 }
        flops { key: "F32xF32->F32" value: 164886643734643 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 345921979381443 }
        flops { key: "F32xF32->F32" value: 145414656554712 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 305040290909090 }
        flops { key: "F32xF32->F32" value: 166626602110490 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 300936609865470 }
        flops { key: "F32xF32->F32" value: 165700898765432 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 459649753424657 }
        flops { key: "F32xF32->F32" value: 220707466392600 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 256
        flops { key: "BF16xBF16->BF16" value: 392736585223116 }
        flops { key: "F32xF32->F32" value: 203978310030395 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 756830766153812 }
        flops { key: "F32xF32->F32" value: 311028078445571 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 742303369512616 }
        flops { key: "F32xF32->F32" value: 325524275882977 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 746431577337504 }
        flops { key: "F32xF32->F32" value: 307443558426800 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 230219087478559 }
        flops { key: "F32xF32->F32" value: 95122415308291 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 349070814044213 }
        flops { key: "F32xF32->F32" value: 148061476006618 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 225576013445378 }
        flops { key: "F32xF32->F32" value: 85163532994923 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 593883752212389 }
        flops { key: "F32xF32->F32" value: 295227336816057 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 47259763380281 }
        flops { key: "F32xF32->F32" value: 32140260536398 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 40920039024390 }
        flops { key: "F32xF32->F32" value: 26757920255183 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 469703335083114 }
        flops { key: "F32xF32->F32" value: 178837745502998 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 485416737793851 }
        flops { key: "F32xF32->F32" value: 188574257815244 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 623905766414875 }
        flops { key: "F32xF32->F32" value: 280057857068335 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 794300190843863 }
        flops { key: "F32xF32->F32" value: 347202429910059 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 677012499369483 }
        flops { key: "F32xF32->F32" value: 311006963929796 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 239674514285714 }
        flops { key: "F32xF32->F32" value: 124506241187384 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 528416251968503 }
        flops { key: "F32xF32->F32" value: 167353775561097 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 801750947052183 }
        flops { key: "F32xF32->F32" value: 335243942736854 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 323416212048192 }
        flops { key: "F32xF32->F32" value: 103883690402476 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 699050666666666 }
        flops { key: "F32xF32->F32" value: 279729536016673 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 700418671885192 }
        flops { key: "F32xF32->F32" value: 245006691158014 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 182857940054495 }
        flops { key: "F32xF32->F32" value: 70492504201680 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 512
        flops { key: "BF16xBF16->BF16" value: 339791716455696 }
        flops { key: "F32xF32->F32" value: 179555488963210 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 629391456037514 }
        flops { key: "F32xF32->F32" value: 168206639062813 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 578213152396338 }
        flops { key: "F32xF32->F32" value: 169306500157678 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 215437765650080 }
        flops { key: "F32xF32->F32" value: 113168404721753 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 404880024132730 }
        flops { key: "F32xF32->F32" value: 107117101356743 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 742303369512616 }
        flops { key: "F32xF32->F32" value: 372313959365891 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 138941747412008 }
        flops { key: "F32xF32->F32" value: 62022979667282 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 447392426666666 }
        flops { key: "F32xF32->F32" value: 229334007689021 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 489845722627737 }
        flops { key: "F32xF32->F32" value: 211449748719968 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 142481664543524 }
        flops { key: "F32xF32->F32" value: 68200065040650 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 681524483655982 }
        flops { key: "F32xF32->F32" value: 277168426823267 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 512
        flops { key: "BF16xBF16->BF16" value: 20971520000000 }
        flops { key: "F32xF32->F32" value: 15477136531365 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 305387321956769 }
        flops { key: "F32xF32->F32" value: 180886425876010 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 171633923273657 }
        flops { key: "F32xF32->F32" value: 72198885422270 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 512
        flops { key: "BF16xBF16->BF16" value: 240103270125223 }
        flops { key: "F32xF32->F32" value: 110649404781533 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 521233895145631 }
        flops { key: "F32xF32->F32" value: 159072862814814 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 590617064906490 }
        flops { key: "F32xF32->F32" value: 175879086650286 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 72628640692640 }
        flops { key: "F32xF32->F32" value: 37786522522522 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 773171430423042 }
        flops { key: "F32xF32->F32" value: 325266194143554 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 25115592814371 }
        flops { key: "F32xF32->F32" value: 17403751037344 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 139810133333333 }
        flops { key: "F32xF32->F32" value: 74565404444444 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 209715200000000 }
        flops { key: "F32xF32->F32" value: 94386587904360 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 729940057104010 }
        flops { key: "F32xF32->F32" value: 225742000210238 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 2048
        flops { key: "BF16xBF16->BF16" value: 763142732054015 }
        flops { key: "F32xF32->F32" value: 326242118577000 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 4096
        flops { key: "BF16xBF16->BF16" value: 740767039668851 }
        flops { key: "F32xF32->F32" value: 300473436127046 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 256
        flops { key: "BF16xBF16->BF16" value: 12520310447761 }
        flops { key: "F32xF32->F32" value: 9731563805104 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 256
        flops { key: "BF16xBF16->BF16" value: 212369822784810 }
        flops { key: "F32xF32->F32" value: 111384006639004 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 1024
        flops { key: "BF16xBF16->BF16" value: 245820014652014 }
        flops { key: "F32xF32->F32" value: 110649404781533 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 256
        flops { key: "BF16xBF16->BF16" value: 25653235474006 }
        flops { key: "F32xF32->F32" value: 18275834422657 }
      }
    }
  }
)pb";

#endif  // XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_DATA_H_
