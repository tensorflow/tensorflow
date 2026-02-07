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

#ifndef XLA_BACKENDS_GPU_COST_MODEL_MATMUL_INTERPOLATOR_DATA_H_
#define XLA_BACKENDS_GPU_COST_MODEL_MATMUL_INTERPOLATOR_DATA_H_

// BEGIN_DEFAULT_PERF_TABLE
constexpr char kDefaultMatmulPTable[] = R"pb(
  entries {
    key: "sm_100"
    value {
      entries {
        b: 1
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 6765006451612 }
        flops { key: "f32xf32->f32" value: 5065584541062 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 6919866364198 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 5531558193208 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 6808935064935 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 13400332268370 }
        flops { key: "f32xf32->f32" value: 11552567395420 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 12404595933456 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 13148288401253 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 13107200000000 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 18551171803731 }
        flops { key: "f32xf32->f32" value: 20161893946222 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 21287506423473 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 20453783602560 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 21236982278481 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 46091252747252 }
        flops { key: "f32xf32->f32" value: 30448667876588 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 41721395088591 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 43690666666666 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 52093044052008 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 89240510638297 }
        flops { key: "f32xf32->f32" value: 67378377510040 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 92182505494505 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 94519526760563 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 92691801104972 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 12593143929442 }
        flops { key: "f32xf32->f32" value: 11244782841823 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 13187043426999 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 12018063037249 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 13357656050955 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 21017495772001 }
        flops { key: "f32xf32->f32" value: 17329596901226 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 26540978445718 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 21076904522613 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 21287506423473 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 39102032920611 }
        flops { key: "f32xf32->f32" value: 34370737003841 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 50676884274117 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 52103155279503 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 52408327996876 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 88738993719008 }
        flops { key: "f32xf32->f32" value: 63310249056603 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 94769799117387 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 93727463687150 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 93190576636000 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 163281907542579 }
        flops { key: "f32xf32->f32" value: 125173912800186 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 157903209411764 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 178897338220593 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 181344675561560 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 20213513253012 }
        flops { key: "f32xf32->f32" value: 15590397026367 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 21024080200501 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 24813778517286 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 21126668975287 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 33156553359683 }
        flops { key: "f32xf32->f32" value: 31007907589234 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 48770976744186 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 41514917414166 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 41323192118226 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 81245598062954 }
        flops { key: "f32xf32->f32" value: 61105271113134 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 81245598062954 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 81024888620585 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 96943104369808 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 156796411214953 }
        flops { key: "f32xf32->f32" value: 119810513724615 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 148429889964058 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 147816881057268 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 175218966057441 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 292333739177783 }
        flops { key: "f32xf32->f32" value: 214042026113824 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 311410041763341 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 343212985136646 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 323318826859379 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 34583284720432 }
        flops { key: "f32xf32->f32" value: 43226321417069 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 44020245326336 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 43464290155440 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 44136049983558 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 69024287991771 }
        flops { key: "f32xf32->f32" value: 49417425625920 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 84720042922518 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 85789535314797 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 85380234096692 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 136677930753564 }
        flops { key: "f32xf32->f32" value: 99855093834278 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 162442030862329 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 160932527577937 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 153919412844036 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 266768155031055 }
        flops { key: "f32xf32->f32" value: 186381153272001 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 309168391592283 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 299551352768865 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 294296786076469 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 464421204152249 }
        flops { key: "f32xf32->f32" value: 305735143507972 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 566170220933298 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 459600566720171 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 468371569901853 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 53081956891437 }
        flops { key: "f32xf32->f32" value: 37241322974472 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 68759081967213 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 68900271047227 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 67361469510664 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 104694015600624 }
        flops { key: "f32xf32->f32" value: 73827133113311 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 135300129032258 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 134739844898983 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 129288600120409 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 205855411042944 }
        flops { key: "f32xf32->f32" value: 138941747412008 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 260585323140395 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 247548547319884 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 253689739870053 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 394106009910075 }
        flops { key: "f32xf32->f32" value: 272800260162601 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 488897814001138 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 509365191650853 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 520097759263744 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 651394145142943 }
        flops { key: "f32xf32->f32" value: 424382915468603 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 930452187175043 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 906876540540540 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 920777638760853 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 13568310553983 }
        flops { key: "f32xf32->f32" value: 11066765171503 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 12863497028943 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 12826617737003 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 12190529336966 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 23489276863843 }
        flops { key: "f32xf32->f32" value: 20311399515738 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 26214400000000 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 21183353535353 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 21290883248730 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 47127011235955 }
        flops { key: "f32xf32->f32" value: 31707471769430 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 41521338901778 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 51781530864197 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 43464290155440 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 90933420054200 }
        flops { key: "f32xf32->f32" value: 67778173462946 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 93466384401114 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 92166680171673 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 92182505494505 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 165241893505694 }
        flops { key: "f32xf32->f32" value: 114495822563446 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 169466828282828 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 171196081632653 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 151830009049773 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 17924376068376 }
        flops { key: "f32xf32->f32" value: 18594863951233 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 25885772034715 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 20971520000000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 19593828905109 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 48480306303052 }
        flops { key: "f32xf32->f32" value: 34304850607028 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 39290903981264 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 47384899558693 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 52748173707997 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 89478485333333 }
        flops { key: "f32xf32->f32" value: 65154236893203 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 91678775956284 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 102573731753916 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 98962380092165 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 162466609774549 }
        flops { key: "f32xf32->f32" value: 124737665427509 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 161683756060834 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 166471600620155 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 181344675561560 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 291698403694648 }
        flops { key: "f32xf32->f32" value: 199710187668557 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 328965019607843 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 326514162688155 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 311319751812119 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 44143308008551 }
        flops { key: "f32xf32->f32" value: 30890156041426 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 41630808933002 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 41425224691358 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 47645625843095 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 80273760765550 }
        flops { key: "f32xf32->f32" value: 60897335753176 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 81616131346913 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 93727463687150 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 91149560611205 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 157163615925058 }
        flops { key: "f32xf32->f32" value: 116495803840729 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 154273250574712 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 153523280526165 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 177477987438016 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 289262344827586 }
        flops { key: "f32xf32->f32" value: 213679964975124 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 312134251162790 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 284887721942159 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 328060441185456 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 492429178628754 }
        flops { key: "f32xf32->f32" value: 311703846142680 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 584667478355567 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 583555339130434 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 581029125541125 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 69741609768771 }
        flops { key: "f32xf32->f32" value: 50595694279521 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 84519979848866 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 84506675901149 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 84733414141414 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 135283082272899 }
        flops { key: "f32xf32->f32" value: 100162483582089 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 151487277652370 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 146846529540481 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 150764086492558 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 263107528546924 }
        flops { key: "f32xf32->f32" value: 187930659665704 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 300852290277388 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 286178524520255 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 307750594439667 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 471611649939606 }
        flops { key: "f32xf32->f32" value: 313867823443437 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 579617718758434 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 533668898608349 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 504577924812030 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 761316546308605 }
        flops { key: "f32xf32->f32" value: 443237079050567 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 972372038940457 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 983280058608058 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 975907133833219 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 103383576352782 }
        flops { key: "f32xf32->f32" value: 73252955655614 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 131328500978473 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 135847902834008 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 134993943173246 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 204893010972235 }
        flops { key: "f32xf32->f32" value: 139664649323621 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 265679036001484 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 244448907000569 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 242680941123290 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 396397535394554 }
        flops { key: "f32xf32->f32" value: 275813466221423 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 477643160142348 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 516097968757510 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 471767057996485 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 669258635917413 }
        flops { key: "f32xf32->f32" value: 434999472932597 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 920777638760853 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 928741982052113 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 864439427593841 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 990479630095128 }
        flops { key: "f32xf32->f32" value: 582605438958220 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1472896877914952 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1493382230876217 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1480640281306558 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 26214400000000 }
        flops { key: "f32xf32->f32" value: 19152073059360 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 21902370757180 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 21672489585015 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 22787390152801 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 42460527681113 }
        flops { key: "f32xf32->f32" value: 35098778242677 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 41630808933002 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 51622203076923 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 52418561999609 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 90443212938005 }
        flops { key: "f32xf32->f32" value: 68618470347648 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 93206755555555 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 92166680171673 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 93466384401114 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 166885580354367 }
        flops { key: "f32xf32->f32" value: 116281332466969 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 175218966057441 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 167327695808009 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 150785258250245 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 274403737285969 }
        flops { key: "f32xf32->f32" value: 181375308108108 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 243148057971014 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 289223386936026 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 291105279652975 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 47926344581324 }
        flops { key: "f32xf32->f32" value: 33884808886644 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 51306470948012 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 46863731843575 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 51463852760736 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 88272099967116 }
        flops { key: "f32xf32->f32" value: 65664250489236 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 101660843022154 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 99827242841204 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 91914211950008 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 164080352078239 }
        flops { key: "f32xf32->f32" value: 122210542226269 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 163281907542579 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 174762666666666 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 164836018421860 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 293652898673594 }
        flops { key: "f32xf32->f32" value: 201207125269371 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 333874945273631 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 329773287469287 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 330534654148068 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 467606673489384 }
        flops { key: "f32xf32->f32" value: 294599581315590 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 547687744963019 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 497102696296296 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 498024964749536 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 77314359447004 }
        flops { key: "f32xf32->f32" value: 61008058181818 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 82620946752847 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 91678775956284 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 92436451790633 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 152824056931397 }
        flops { key: "f32xf32->f32" value: 114716006837606 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 156044444702804 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 154273250574712 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 170760468193384 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 288020875536480 }
        flops { key: "f32xf32->f32" value: 208736746500777 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 332994828345479 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 308457863832232 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 325771184466019 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 492542121100917 }
        flops { key: "f32xf32->f32" value: 316178393404004 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 586023645244917 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 498950661710037 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 563940033613445 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 723362913010526 }
        flops { key: "f32xf32->f32" value: 425412767036450 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 863136514469453 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 967117157396982 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 856082777755630 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 135283082272899 }
        flops { key: "f32xf32->f32" value: 99864380952380 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 131296383467840 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 150426145138694 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 158978653242522 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 263624312300515 }
        flops { key: "f32xf32->f32" value: 183859901369863 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 304348589569161 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 294296786076469 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 302249633779028 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 468473745200698 }
        flops { key: "f32xf32->f32" value: 323002729638264 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 575887274872620 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 465932663918420 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 563940033613445 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 760305770224818 }
        flops { key: "f32xf32->f32" value: 454493893756613 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 979468026453819 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 990194189279538 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 983280058608058 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 981258235320996 }
        flops { key: "f32xf32->f32" value: 559714249820811 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1431655765333333 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1437044682894186 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1478983228650137 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 204288779299847 }
        flops { key: "f32xf32->f32" value: 144165121374865 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 250406208955223 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 260111875968992 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 255652815238095 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 394069850077988 }
        flops { key: "f32xf32->f32" value: 269158820329635 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 487068189612156 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 475949390070922 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 511183920019043 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 675256237088279 }
        flops { key: "f32xf32->f32" value: 442233041186161 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 923847557754355 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 919102781082816 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 925440055160525 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1014878850661625 }
        flops { key: "f32xf32->f32" value: 584508341861731 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1470879210958904 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1492992889893108 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1490919828516879 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1213953447145279 }
        flops { key: "f32xf32->f32" value: 656584784697406 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1937615652625049 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1934452109447134 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1951814267666439 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 41020088019559 }
        flops { key: "f32xf32->f32" value: 33091155818540 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 52093044052008 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 51140304057915 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 43351979328165 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 90687654054054 }
        flops { key: "f32xf32->f32" value: 68200065040650 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 88504931091328 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 88069375328083 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 89225679242147 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 165267327074034 }
        flops { key: "f32xf32->f32" value: 115879756529246 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 174705796290270 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 177009862182657 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 169440085845037 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 276097151967086 }
        flops { key: "f32xf32->f32" value: 181084716080613 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 293612749248017 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 289887101511879 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 301570516500491 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 406105077155824 }
        flops { key: "f32xf32->f32" value: 275247840041015 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 488953471766848 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 489845722627737 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 421281735752820 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 86466566596875 }
        flops { key: "f32xf32->f32" value: 65027968992248 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 89717732620320 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 97506522339266 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 93433851722937 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 162098705314009 }
        flops { key: "f32xf32->f32" value: 120672266127219 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 179916525469168 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 179435465240641 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 164030220592728 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 231784527576902 }
        flops { key: "f32xf32->f32" value: 203668783004552 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 329672036843721 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 330534654148068 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 315759983531833 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 470114633975481 }
        flops { key: "f32xf32->f32" value: 296531848660591 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 503631249530956 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 509365191650853 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 497045167920379 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 647466238938720 }
        flops { key: "f32xf32->f32" value: 405798119425548 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 818322815280556 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 867144618614980 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 874204619580704 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 150089715403969 }
        flops { key: "f32xf32->f32" value: 112410157453936 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 166059669656665 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 169013351802298 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 169013351802298 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 284359593220339 }
        flops { key: "f32xf32->f32" value: 214405316293929 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 314235242610477 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 311319751812119 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 303574165677127 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 496069218757218 }
        flops { key: "f32xf32->f32" value: 318357964272477 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 504518653353694 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 582289492407809 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 541064159234064 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 725378702246242 }
        flops { key: "f32xf32->f32" value: 426744229320880 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 970614078192090 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 979468026453819 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 969081068592057 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 911350548193729 }
        flops { key: "f32xf32->f32" value: 538216453132832 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1423942742146705 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1373071386189258 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1394017298279779 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 261060496960855 }
        flops { key: "f32xf32->f32" value: 184317539095356 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 305735143507972 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 256630455066921 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 297559047803796 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 466033777777777 }
        flops { key: "f32xf32->f32" value: 327360312195121 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 567516820295983 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 497102696296296 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 544355804309252 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 760373071788970 }
        flops { key: "f32xf32->f32" value: 452292259477674 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 985084242201834 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 992138437514437 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 990422528767439 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 985762519164562 }
        flops { key: "f32xf32->f32" value: 556342913989637 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1427729509183079 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1472896877914952 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1425715284979253 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1172847432004369 }
        flops { key: "f32xf32->f32" value: 630003087111974 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2035167823728010 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2043276544243577 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2055008275598086 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 397093869822485 }
        flops { key: "f32xf32->f32" value: 270583210231210 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 502688119850187 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 496069218757218 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 486296115942029 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 666920387577639 }
        flops { key: "f32xf32->f32" value: 457178912768109 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 855997468061783 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 888675211255948 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 884466082372322 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1017704471536046 }
        flops { key: "f32xf32->f32" value: 574078366103054 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1484735043125054 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1464609478601875 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1514045050145413 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1208488265616207 }
        flops { key: "f32xf32->f32" value: 657577477761616 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1955813887067395 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1964311592042076 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1967911704925544 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1342544381979447 }
        flops { key: "f32xf32->f32" value: 700304466981901 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2491280334106728 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2489836113623188 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2492183823021687 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 89717732620320 }
        flops { key: "f32xf32->f32" value: 66841498007968 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 84519979848866 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 92420539163367 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 85163532994923 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 161708106024096 }
        flops { key: "f32xf32->f32" value: 115904773747841 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 169413351846008 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 166059669656665 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 168562295761381 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 271112693851786 }
        flops { key: "f32xf32->f32" value: 170719743063836 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 225576013445378 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 291777669565217 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 290475266874070 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 403586477729750 }
        flops { key: "f32xf32->f32" value: 277829568277378 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 487123431552682 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 488842168905076 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 473431139329806 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 514706368985559 }
        flops { key: "f32xf32->f32" value: 359276196913296 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 666920387577639 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 634552307896875 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 617093002298850 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 160884300869044 }
        flops { key: "f32xf32->f32" value: 123333542844015 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 176138750656167 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 177978091165257 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 175218966057441 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 287943637436310 }
        flops { key: "f32xf32->f32" value: 200306281876690 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 322590303139552 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 324148475169811 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 308546501149425 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 465175706270984 }
        flops { key: "f32xf32->f32" value: 282210874301859 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 538891756085319 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 496183837338262 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 505349723026238 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 652334036452004 }
        flops { key: "f32xf32->f32" value: 407956620060790 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 871278485850491 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 857450049111599 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 801150400298451 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 805318951108611 }
        flops { key: "f32xf32->f32" value: 517965182826821 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1172127255509313 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1034433356454720 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1009037306707388 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 283683440951122 }
        flops { key: "f32xf32->f32" value: 210992694831990 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 312816263364894 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 323318826859379 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 305040290909090 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 490573077784123 }
        flops { key: "f32xf32->f32" value: 315065089201877 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 510333566539923 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 569926658174097 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 515107615255456 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 715648970424060 }
        flops { key: "f32xf32->f32" value: 419102975800156 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 850825534072900 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 915966580507570 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 919299506849315 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 913822828936170 }
        flops { key: "f32xf32->f32" value: 546155556459816 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1332185885856079 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1405076403369592 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1441022411004865 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1075624166291009 }
        flops { key: "f32xf32->f32" value: 636857546856465 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1895395982347749 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1751617983686786 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1746986900955867 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 458766000427259 }
        flops { key: "f32xf32->f32" value: 308457863832232 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 471767057996485 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 472597633802816 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 562757769392033 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 762397673914972 }
        flops { key: "f32xf32->f32" value: 441052299856233 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 979468026453819 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 893296026622296 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 890333187396351 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 990308345861194 }
        flops { key: "f32xf32->f32" value: 556757597433321 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1462614437595777 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1416195629709010 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1445143773889636 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1170011862566826 }
        flops { key: "f32xf32->f32" value: 629783686498772 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1833497244823906 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1900428007079646 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1896651488628836 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1277788708367422 }
        flops { key: "f32xf32->f32" value: 740001257064093 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2484974207564909 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2481116248546774 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2484075937536148 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 645132151107773 }
        flops { key: "f32xf32->f32" value: 423733947908445 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 861751062600321 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 913045768707483 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 916161965870307 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 991451361034164 }
        flops { key: "f32xf32->f32" value: 584687376510227 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1484991717866712 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1470879210958904 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1485120088520055 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1204928403983728 }
        flops { key: "f32xf32->f32" value: 655520039072039 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1951814267666439 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1924268501792114 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1958936052907639 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1343541814655509 }
        flops { key: "f32xf32->f32" value: 698028164472615 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2499435394486069 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2500799764765821 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2502894694638694 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1416867213789406 }
        flops { key: "f32xf32->f32" value: 757940978271899 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2764981863158106 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2785491852050019 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2784419640842787 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 13527285627897 }
        flops { key: "f32xf32->f32" value: 11212842773600 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 12942886017357 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 9508198356474 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 11911406460773 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 24813778517286 }
        flops { key: "f32xf32->f32" value: 19551016460305 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 26132735202492 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 21895224796084 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 21952523388943 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 46337900224408 }
        flops { key: "f32xf32->f32" value: 31293478200046 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 50830421511077 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 43690666666666 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 53071462238038 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 88054930621617 }
        flops { key: "f32xf32->f32" value: 67633019904258 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 95021400353982 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 82620946752847 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 92182505494505 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 164836018421860 }
        flops { key: "f32xf32->f32" value: 111476518272425 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 165700898765432 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 165241893505694 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 167353775561097 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 22129880956306 }
        flops { key: "f32xf32->f32" value: 18152248850419 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 26296576802507 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 19284156321839 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 26291425661116 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 45582519273221 }
        flops { key: "f32xf32->f32" value: 34309235173824 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 46210269581683 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 51771544069431 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 39568905660377 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 88301136842105 }
        flops { key: "f32xf32->f32" value: 65408249512670 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 91647475588938 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 94753073067419 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 92166680171673 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 162442030862329 }
        flops { key: "f32xf32->f32" value: 118527632630533 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 164080352078239 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 159403477434679 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 174308737662337 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 293693059080962 }
        flops { key: "f32xf32->f32" value: 190870469113856 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 324099554482342 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 323416212048192 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 325672376099484 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 40524676328502 }
        flops { key: "f32xf32->f32" value: 30720468757152 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 40920039024390 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 48489063583815 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 47798336182336 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 81245598062954 }
        flops { key: "f32xf32->f32" value: 61447969783678 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 89003798408488 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 81430443197330 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 81430443197330 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 157140615249524 }
        flops { key: "f32xf32->f32" value: 115480944719294 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 149421350403562 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 173857160621761 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 173800877954030 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 292333739177783 }
        flops { key: "f32xf32->f32" value: 200324967164179 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 313592822429906 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 307838825688073 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 337124591522762 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 488842168905076 }
        flops { key: "f32xf32->f32" value: 306695750928306 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 573580034188034 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 569851040997744 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 574654441530639 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 68460968120377 }
        flops { key: "f32xf32->f32" value: 50748739200302 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 76948675935215 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 80829706714844 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 81628540672038 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 135283082272899 }
        flops { key: "f32xf32->f32" value: 99420539259259 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 150131686800894 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 154628718894009 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 160932527577937 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 261601126568400 }
        flops { key: "f32xf32->f32" value: 187717102097902 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 305691622491103 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 309882200288600 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 309882200288600 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 460438174957118 }
        flops { key: "f32xf32->f32" value: 302292180180180 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 570987409731454 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 571139268085106 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 571139268085106 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 732242314551189 }
        flops { key: "f32xf32->f32" value: 452649765084049 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 968862462440785 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 968862462440785 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 967334976576576 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 104034669508768 }
        flops { key: "f32xf32->f32" value: 73168097035775 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 135300129032258 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 136677930753564 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 131586007843137 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 205186666157080 }
        flops { key: "f32xf32->f32" value: 142604664851583 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 249417380720092 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 246242821694759 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 256110154800238 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 395922501474926 }
        flops { key: "f32xf32->f32" value: 271403936556082 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 472493651925192 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 498024964749536 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 506422272845183 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 652334036452004 }
        flops { key: "f32xf32->f32" value: 416501871217998 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 927038052233973 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 820903535168195 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 928641577513513 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 993056022196531 }
        flops { key: "f32xf32->f32" value: 552762843758043 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1431655765333333 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1418417204755614 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1431536470627447 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 20164923076923 }
        flops { key: "f32xf32->f32" value: 19642576906190 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 21732145077720 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 21841778356387 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 26630501587301 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 46725057615317 }
        flops { key: "f32xf32->f32" value: 30448667876588 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 42581766497461 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 51781530864197 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 43233283298437 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 75551774838164 }
        flops { key: "f32xf32->f32" value: 67378377510040 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 91180521739130 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 93727463687150 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 91914211950008 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 164482509803921 }
        flops { key: "f32xf32->f32" value: 114495822563446 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 153919412844036 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 151103549676329 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 174734226851098 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 273913730612244 }
        flops { key: "f32xf32->f32" value: 176326763116840 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 282489298605630 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 265711908933432 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 296941876106194 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 39850869358669 }
        flops { key: "f32xf32->f32" value: 33825032258064 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 46995002801120 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 47118738985430 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 51130563047619 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 89210852775008 }
        flops { key: "f32xf32->f32" value: 64902189555125 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 87381333333333 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 93174403332176 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 86452642834138 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 162466609774549 }
        flops { key: "f32xf32->f32" value: 121574028985507 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 163232262693827 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 168562295761381 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 179405484377610 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 292971848294679 }
        flops { key: "f32xf32->f32" value: 202688404719207 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 312134251162790 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 309971658198614 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 312134251162790 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 466844271304347 }
        flops { key: "f32xf32->f32" value: 287365669476783 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 485307039096045 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 538891756085319 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 549861387274356 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 79324898345153 }
        flops { key: "f32xf32->f32" value: 61780312082853 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 91165038546442 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 88069375328083 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 91165038546442 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 151146090090090 }
        flops { key: "f32xf32->f32" value: 113551377326565 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 167353775561097 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 168986752282027 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 149442146694502 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 289184439536762 }
        flops { key: "f32xf32->f32" value: 203978310030395 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 325771184466019 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 309257437788018 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 306389448994150 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 495268369003690 }
        flops { key: "f32xf32->f32" value: 316924977567886 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 562757769392033 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 577202969493347 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 580871963213416 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 718702693440428 }
        flops { key: "f32xf32->f32" value: 422961967206657 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 836084737395367 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 931865327836841 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 917532000854518 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 133649716704007 }
        flops { key: "f32xf32->f32" value: 99864380952380 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 156044444702804 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 156067125581395 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 158626358989511 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 261123984435797 }
        flops { key: "f32xf32->f32" value: 176834951251646 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 295633762114537 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 302207099352659 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 301570516500491 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 466793532876861 }
        flops { key: "f32xf32->f32" value: 314995767950128 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 542156942186316 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 577125409298575 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 462819751724137 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 757222724964739 }
        flops { key: "f32xf32->f32" value: 458741500240320 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 974136379224313 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 975907133833219 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 976128930909090 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 972372038940457 }
        flops { key: "f32xf32->f32" value: 558259218301163 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1445143773889636 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1416545941952506 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1431417195800700 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 205225883792048 }
        flops { key: "f32xf32->f32" value: 145100246486486 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 242708368896925 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 259107583011583 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 254682595825426 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 391876578102189 }
        flops { key: "f32xf32->f32" value: 271971080040526 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 478334702750863 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 477643160142348 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 474162872157209 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 671088640000000 }
        flops { key: "f32xf32->f32" value: 438239609815825 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 923748208624583 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 872694767042568 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 923847557754355 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 989623800921659 }
        flops { key: "f32xf32->f32" value: 581048776811986 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1490919828516879 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1499639418994413 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1495462150417827 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1209168720720720 }
        flops { key: "f32xf32->f32" value: 662241507362578 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1908874353777777 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1903797560283688 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1908874353777777 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 41721395088591 }
        flops { key: "f32xf32->f32" value: 34450135523613 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 51931796479009 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 48620803477630 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 51306470948012 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 90443212938005 }
        flops { key: "f32xf32->f32" value: 67378377510040 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 92420539163367 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 88989045582628 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 92436451790633 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 165292768472906 }
        flops { key: "f32xf32->f32" value: 113743837288135 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 174280445382243 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 174762666666666 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 169413351846008 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 275036327868852 }
        flops { key: "f32xf32->f32" value: 178956970666666 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 293011822622458 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 240103270125223 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 300263373601789 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 404880024132730 }
        flops { key: "f32xf32->f32" value: 274754816786079 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 467555769214021 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 481822671752299 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 470836142951107 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 84519979848866 }
        flops { key: "f32xf32->f32" value: 64265131912856 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 92948565096952 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 89448669110296 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 89928125963149 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 163680156097560 }
        flops { key: "f32xf32->f32" value: 122657279415124 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 174762666666666 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 167772160000000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 162836188049742 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 293612749248017 }
        flops { key: "f32xf32->f32" value: 197615132787337 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 309926922788281 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 310599312698871 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 310599312698871 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 466033777777777 }
        flops { key: "f32xf32->f32" value: 289242864570004 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 553332555526926 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 500812417910447 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 510333566539923 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 648394821256038 }
        flops { key: "f32xf32->f32" value: 409395414736440 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 868723158576051 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 806112480480480 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 806112480480480 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 148470938053097 }
        flops { key: "f32xf32->f32" value: 113144554689146 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 171633923273657 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 167353775561097 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 168166299765074 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 288020875536480 }
        flops { key: "f32xf32->f32" value: 209326800662832 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 322638769230769 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 320999050523168 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 322638769230769 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 496183837338262 }
        flops { key: "f32xf32->f32" value: 317229285471600 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 579617718758434 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 580871963213416 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 583396807389296 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 715768235313723 }
        flops { key: "f32xf32->f32" value: 426659444295435 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 915966580507570 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 887389937190082 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 894598478650281 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 906876540540540 }
        flops { key: "f32xf32->f32" value: 528920574612850 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1380130879177378 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1335499781094527 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1379798344229379 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 263107528546924 }
        flops { key: "f32xf32->f32" value: 177536677248677 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 306389448994150 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 296286375275938 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 296204641103448 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 466844271304347 }
        flops { key: "f32xf32->f32" value: 323318826859379 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 457300606473594 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 563940033613445 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 573426875300400 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 758292248587570 }
        flops { key: "f32xf32->f32" value: 456425855047821 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 974136379224313 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 977684337810152 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 976128930909090 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 983055000228885 }
        flops { key: "f32xf32->f32" value: 551768665981500 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1387263338501292 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1474920087912088 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1486790928948507 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1159939854432516 }
        flops { key: "f32xf32->f32" value: 630349820543396 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1807646168350168 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1803092903442485 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1900007651404556 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 394178349486049 }
        flops { key: "f32xf32->f32" value: 238397385435168 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 495268369003690 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 493334171376062 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 504577924812030 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 681146189199904 }
        flops { key: "f32xf32->f32" value: 317287873231633 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 914601212947189 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 808540530120481 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 849479291139240 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 991222546965151 }
        flops { key: "f32xf32->f32" value: 574750566525041 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1470753290300488 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1470501513652315 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1482684835073789 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1208488265616207 }
        flops { key: "f32xf32->f32" value: 655682658779077 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1936087134050825 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1913551925150367 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1937615652625049 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1340475504457232 }
        flops { key: "f32xf32->f32" value: 708857451064532 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2324745491745602 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2363361995253981 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2277213663916227 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 89225679242147 }
        flops { key: "f32xf32->f32" value: 68182742189484 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 91678775956284 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 84720042922518 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 85584395345129 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 162049777241171 }
        flops { key: "f32xf32->f32" value: 115084868595927 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 169013351802298 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 166523235732009 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 172933133193751 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 271695805668016 }
        flops { key: "f32xf32->f32" value: 178451358484294 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 222906751920282 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 238821580071174 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 294256460400109 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 404880024132730 }
        flops { key: "f32xf32->f32" value: 267849535141877 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 470114633975481 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 466844271304347 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 470887764060958 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 534731984063745 }
        flops { key: "f32xf32->f32" value: 354282545244576 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 636857546856465 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 632170635266411 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 642814831400134 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 161319384615384 }
        flops { key: "f32xf32->f32" value: 123376056991841 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 177536677248677 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 176138750656167 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 171606492568323 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 288020875536480 }
        flops { key: "f32xf32->f32" value: 201528120120120 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 318759633071099 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 320328706443914 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 319470938411187 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 460339474383708 }
        flops { key: "f32xf32->f32" value: 290514562770562 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 543322871094244 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 496987652858134 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 537812083145504 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 646783720502974 }
        flops { key: "f32xf32->f32" value: 412026793553338 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 805885598273759 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 867319728594507 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 807324679699248 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 804075127960310 }
        flops { key: "f32xf32->f32" value: 518465390632544 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1014699024511251 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1163317252437703 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1056832503937007 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 289808859379217 }
        flops { key: "f32xf32->f32" value: 211699886435331 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 329773287469287 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 307838825688073 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 302292180180180 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 498776831494599 }
        flops { key: "f32xf32->f32" value: 315806418823529 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 513199581311984 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 556775641171895 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 573580034188034 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 719485266102688 }
        flops { key: "f32xf32->f32" value: 427444993630573 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 960305711794298 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 965595165467625 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 920876349914236 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 910529424634301 }
        flops { key: "f32xf32->f32" value: 532214039157373 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1342177280000000 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1385473321290322 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1429392560445960 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1074547734801100 }
        flops { key: "f32xf32->f32" value: 621198625397743 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1753942744665645 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1747342268510984 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1915151795775040 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 469190222416430 }
        flops { key: "f32xf32->f32" value: 330509218622547 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 566170220933298 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 481067125448028 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 562684042447268 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 759163463720724 }
        flops { key: "f32xf32->f32" value: 458447701980039 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 986895058823529 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 988484993325661 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 990422528767439 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 979635581000171 }
        flops { key: "f32xf32->f32" value: 553760610624033 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1464609478601875 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1458516782748960 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1462614437595777 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1160566721880699 }
        flops { key: "f32xf32->f32" value: 632939217625170 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2047172209723546 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1918573810262996 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2045222521904762 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1277123787094855 }
        flops { key: "f32xf32->f32" value: 701792041830065 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2489836113623188 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2488393566628041 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2341857849509269 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 655570067312829 }
        flops { key: "f32xf32->f32" value: 413932854279105 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 836247526479750 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 911495606112054 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 917434005340168 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 992310355455438 }
        flops { key: "f32xf32->f32" value: 576660485499463 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1480767900706774 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1484735043125054 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1474793474461327 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1204421563656758 }
        flops { key: "f32xf32->f32" value: 651740105614567 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1957150738664844 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1927723202872531 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1910572640569395 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1341338943160524 }
        flops { key: "f32xf32->f32" value: 710330222713788 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2460594268690919 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2463946817353890 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2483537287170220 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1417378628523105 }
        flops { key: "f32xf32->f32" value: 760667656280094 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2782615676060900 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2778115974126779 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2770443941059082 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 159783009523809 }
        flops { key: "f32xf32->f32" value: 114716006837606 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 172046438711744 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 176080981305346 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 161319384615384 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 266238984378874 }
        flops { key: "f32xf32->f32" value: 175893492341715 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 294256460400109 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 297517823219728 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 240479691825307 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 403056240240240 }
        flops { key: "f32xf32->f32" value: 270310736736106 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 473326790390125 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 463569055153804 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 469241483229542 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 535131733864938 }
        flops { key: "f32xf32->f32" value: 355779265738899 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 643585419345171 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 639132038095238 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 643681872761333 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 793600756836659 }
        flops { key: "f32xf32->f32" value: 460833400858369 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 903775537061392 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 900790120805369 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 900742892256068 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 284359593220339 }
        flops { key: "f32xf32->f32" value: 198546934911242 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 302974555304740 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 322541851607089 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 325672376099484 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 459502224884989 }
        flops { key: "f32xf32->f32" value: 279583862517901 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 538959379595934 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 541132329091596 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 539027020080321 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 642959176047904 }
        flops { key: "f32xf32->f32" value: 400351164802386 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 871455269554631 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 865833544199173 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 798915047619047 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 804715405124361 }
        flops { key: "f32xf32->f32" value: 519217516441005 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1103537331963001 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1071531789683777 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1107806885736394 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 916773082739667 }
        flops { key: "f32xf32->f32" value: 619831481906411 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1535013329521086 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1390857284974093 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1541210117879250 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 481930800718132 }
        flops { key: "f32xf32->f32" value: 308192257175660 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 566170220933298 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 514244168582375 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 571063328812657 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 718702693440428 }
        flops { key: "f32xf32->f32" value: 419409920999951 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 909950698305084 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 920580279927124 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 916064262770608 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 909180206604572 }
        flops { key: "f32xf32->f32" value: 530947528633680 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1409110005249343 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1403354777323966 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1416195629709010 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1074816640640640 }
        flops { key: "f32xf32->f32" value: 618359039124644 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1870223076856085 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1766930904453358 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1880355626771739 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1161174646187120 }
        flops { key: "f32xf32->f32" value: 735439605479452 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2393989783522034 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2398836762523126 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2396745142857143 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 725317452672464 }
        flops { key: "f32xf32->f32" value: 432480847447387 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 954967714508060 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 963645343504599 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 967226054723567 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 961057797270082 }
        flops { key: "f32xf32->f32" value: 560700691383812 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1468867064295485 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1448677728644911 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1394469901298701 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1162687410936654 }
        flops { key: "f32xf32->f32" value: 620447071416963 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1798562519262981 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1893307161560502 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1799598720368721 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1276412138935324 }
        flops { key: "f32xf32->f32" value: 731097151295281 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2484075937536148 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2489204793566849 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2363768462300495 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1344396058651485 }
        flops { key: "f32xf32->f32" value: 813209750260342 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2901024853765619 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2887676299443219 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2898394176848942 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 985027761252221 }
        flops { key: "f32xf32->f32" value: 507919500473036 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1420293417989418 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1438970532205377 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1420293417989418 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1203409161109554 }
        flops { key: "f32xf32->f32" value: 661769580092833 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1884894309506829 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1893724557319224 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1911848340084576 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1335473827389859 }
        flops { key: "f32xf32->f32" value: 686693481118783 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2462181180078825 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2472635173287277 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2460858611853178 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1418402566353963 }
        flops { key: "f32xf32->f32" value: 756222782991460 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2788939802597402 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2793474664065040 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2781545677521199 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1466617794742374 }
        flops { key: "f32xf32->f32" value: 859296898417253 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 3123541589327515 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 3120031633511538 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 3121732444595868 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 24745156342182 }
        flops { key: "f32xf32->f32" value: 18636174396001 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 21620123711340 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 25575024390243 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 21509251282051 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 46329902657921 }
        flops { key: "f32xf32->f32" value: 29852697508896 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 51453988115775 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 50984891927825 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 51140304057915 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 88301136842105 }
        flops { key: "f32xf32->f32" value: 66296729068905 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 90184933982865 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 90687654054054 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 90687654054054 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 162049777241171 }
        flops { key: "f32xf32->f32" value: 112386625915846 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 166523235732009 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 159735469205593 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 172488646425702 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 270566164545798 }
        flops { key: "f32xf32->f32" value: 173590142106539 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 277309355371900 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 292333739177783 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 295593069236063 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 38746457274826 }
        flops { key: "f32xf32->f32" value: 29746836879432 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 51306470948012 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 46466237839709 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 51921751644100 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 74214944982029 }
        flops { key: "f32xf32->f32" value: 64403900191938 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 91397839972761 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 93190576636000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 93206755555555 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 160164353221957 }
        flops { key: "f32xf32->f32" value: 117734849122807 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 172516359897172 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 177477987438016 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 157903209411764 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 289223386936026 }
        flops { key: "f32xf32->f32" value: 192513101568803 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 322590303139552 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 311998205433677 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 308546501149425 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 454205509306260 }
        flops { key: "f32xf32->f32" value: 286121330757444 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 534598866816031 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 536870912000000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 537946805611222 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 78192675793766 }
        flops { key: "f32xf32->f32" value: 61008058181818 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 89958262734584 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 89478485333333 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 77649828174717 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 150447222082107 }
        flops { key: "f32xf32->f32" value: 111824809831285 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 173800877954030 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 164886643734643 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 148470938053097 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 283122432168754 }
        flops { key: "f32xf32->f32" value: 194747768930806 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 301528172985116 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 322638769230769 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 299593142857142 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 489734013226909 }
        flops { key: "f32xf32->f32" value: 291085550389698 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 572357049040511 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 575964502614992 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 572204542499333 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 686425970273293 }
        flops { key: "f32xf32->f32" value: 425328510200039 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 960413080500894 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 912754711720327 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 854719859900497 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 135300129032258 }
        flops { key: "f32xf32->f32" value: 100584714192037 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 148758911609864 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 153545234377234 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 153194724497075 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 260111875968992 }
        flops { key: "f32xf32->f32" value: 177302150594451 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 309257437788018 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 315713561893560 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 250815656155103 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 462023160068846 }
        flops { key: "f32xf32->f32" value: 293632822588364 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 465932663918420 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 482797582733812 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 474162872157209 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 719666101876675 }
        flops { key: "f32xf32->f32" value: 452578218756585 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 970614078192090 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 958484109796920 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 977573072948674 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 961057797270082 }
        flops { key: "f32xf32->f32" value: 549914189174482 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1410614104934723 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1264713573616018 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1394469901298701 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 202401851837888 }
        flops { key: "f32xf32->f32" value: 143548372192513 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 267832832127712 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 244476735883424 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 252259326676847 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 390096938782924 }
        flops { key: "f32xf32->f32" value: 263948334316617 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 510212318365407 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 495154172930597 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 500812417910447 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 644502895558223 }
        flops { key: "f32xf32->f32" value: 387038595656483 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 923847557754355 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 922260531672750 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 922458611683848 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 993343115582538 }
        flops { key: "f32xf32->f32" value: 540910839835017 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1431417195800700 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1418417204755614 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1442837757957504 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1190401135254989 }
        flops { key: "f32xf32->f32" value: 652135939265107 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1869002304612706 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1858890844405972 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1870630355400696 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 46329902657921 }
        flops { key: "f32xf32->f32" value: 35696204255319 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 51612277638915 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 52914538931598 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 43577184415584 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 89003798408488 }
        flops { key: "f32xf32->f32" value: 67769617773289 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 89003798408488 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 93711103508465 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 88054930621617 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 162098705314009 }
        flops { key: "f32xf32->f32" value: 113156478448730 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 168588761815041 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 172960989690721 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 151487277652370 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 272212403092914 }
        flops { key: "f32xf32->f32" value: 172474793028672 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 291105279652975 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 240479691825307 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 285493704865727 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 404880024132730 }
        flops { key: "f32xf32->f32" value: 276168164609053 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 449499455363683 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 468371569901853 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 462769884279711 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 85380234096692 }
        flops { key: "f32xf32->f32" value: 64776895752895 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 89210852775008 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 89717732620320 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 89702742188805 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 165292768472906 }
        flops { key: "f32xf32->f32" value: 120266781362007 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 172046438711744 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 167301624181988 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 178481021276595 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 289184439536762 }
        flops { key: "f32xf32->f32" value: 193676375180375 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 304262347407197 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 319518471656003 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 302207099352659 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 464320788756756 }
        flops { key: "f32xf32->f32" value: 283458770855332 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 537946805611222 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 532477968757748 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 495268369003690 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 630731668404435 }
        flops { key: "f32xf32->f32" value: 399364665581849 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 857621265175718 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 871278485850491 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 813209750260342 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 149462948775055 }
        flops { key: "f32xf32->f32" value: 112598765100671 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 169013351802298 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 171141508447561 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 168192641604010 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 286790017094017 }
        flops { key: "f32xf32->f32" value: 200868361051351 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 322541851607089 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 324884061724659 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 316457949896846 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 494242496662830 }
        flops { key: "f32xf32->f32" value: 310262753449396 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 572357049040511 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 497102696296296 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 573580034188034 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 706234859163035 }
        flops { key: "f32xf32->f32" value: 424970790679265 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 906876540540540 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 909757952976064 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 878496071998363 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 903062930193439 }
        flops { key: "f32xf32->f32" value: 527103033902985 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1410961660972404 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1327245765142150 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1423942742146705 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 258608339113680 }
        flops { key: "f32xf32->f32" value: 180855958228061 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 299509574337517 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 298884293389004 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 297600283813747 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 468473745200698 }
        flops { key: "f32xf32->f32" value: 309591818352194 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 461229305841924 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 457203246327443 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 462023160068846 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 742495858933356 }
        flops { key: "f32xf32->f32" value: 442506418297960 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 970833475587703 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 981258235320996 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 976128930909090 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 959286905131498 }
        flops { key: "f32xf32->f32" value: 553368201507440 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1403240152250265 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1311040078144078 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1312342004736078 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1157673125606469 }
        flops { key: "f32xf32->f32" value: 636315018482166 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1883241346560701 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1880355626771739 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1878813340332458 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 390167813953488 }
        flops { key: "f32xf32->f32" value: 265514793273986 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 494356272559852 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 477590047370176 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 486186019470228 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 653873379919311 }
        flops { key: "f32xf32->f32" value: 417149115773115 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 863136514469453 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 911302205813706 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 931865327836841 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 995126806302131 }
        flops { key: "f32xf32->f32" value: 571443227248536 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1454934720867208 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1460749016580222 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1460749016580222 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1207808575928009 }
        flops { key: "f32xf32->f32" value: 655395001869301 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1912273951914514 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1903481157165808 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1877171020979021 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1339848246914543 }
        flops { key: "f32xf32->f32" value: 665299752505058 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2244634222962600 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2247497276818419 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2265875650751780 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 89687756765786 }
        flops { key: "f32xf32->f32" value: 67108864000000 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 93190576636000 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 86452642834138 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 90443212938005 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 164457317200183 }
        flops { key: "f32xf32->f32" value: 114118591136146 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 171579070629594 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 168139966176010 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 169039959697733 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 271078471093158 }
        flops { key: "f32xf32->f32" value: 176138750656167 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 280790225941422 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 280167468754076 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 297517823219728 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 406643372088619 }
        flops { key: "f32xf32->f32" value: 270055790744466 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 403662339849624 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 443694968595041 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 466033777777777 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 531423817866864 }
        flops { key: "f32xf32->f32" value: 358391797062750 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 630731668404435 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 642094079234564 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 630870636897767 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 162860886394661 }
        flops { key: "f32xf32->f32" value: 120916872072072 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 155705020881670 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 158649796690307 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 154985829099307 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 291065823800487 }
        flops { key: "f32xf32->f32" value: 194465602463098 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 312770703175065 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 319470938411187 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 303660018099547 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 464270597340828 }
        flops { key: "f32xf32->f32" value: 290122081599567 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 532477968757748 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 522247968871595 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 528416251968503 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 648248025960304 }
        flops { key: "f32xf32->f32" value: 408480412382899 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 769018316204118 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 870131137763371 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 862963089411292 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 803698970059880 }
        flops { key: "f32xf32->f32" value: 502452889096864 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1081038836143971 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1033313435823409 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1080223162977867 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 286790017094017 }
        flops { key: "f32xf32->f32" value: 205835679861976 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 309257437788018 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 322638769230769 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 305040290909090 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 502570476948279 }
        flops { key: "f32xf32->f32" value: 307112427314980 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 498776831494599 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 582289492407809 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 515169400983567 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 713924085106383 }
        flops { key: "f32xf32->f32" value: 422068327044025 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 909950698305084 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 958698057142857 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 857621265175718 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 910722497031382 }
        flops { key: "f32xf32->f32" value: 518121394052717 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1418300106001816 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1422174601324503 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1247086903600464 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1062387556984725 }
        flops { key: "f32xf32->f32" value: 614093122104661 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1734639457189014 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1890390535211267 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1733151998385876 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 470011741737798 }
        flops { key: "f32xf32->f32" value: 295633762114537 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 558077871101871 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 573426875300400 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 536870912000000 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 736259071912231 }
        flops { key: "f32xf32->f32" value: 455264712317150 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 986668342752125 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 981482471663619 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 965378129017756 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 971492263288848 }
        flops { key: "f32xf32->f32" value: 552229803407264 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1414330220136659 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1468867064295485 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1451002464864864 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1152663234861954 }
        flops { key: "f32xf32->f32" value: 618381296666906 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1828910329908979 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1810312875026343 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1906755736293007 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1278668417021751 }
        flops { key: "f32xf32->f32" value: 701040313552665 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2407492878923767 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2384768071071627 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2386921734491143 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 658585800199340 }
        flops { key: "f32xf32->f32" value: 396489018786060 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 928842408304498 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 916161965870307 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 917532000854518 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 990536738007380 }
        flops { key: "f32xf32->f32" value: 562334103106281 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1466609969609014 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1472518143824462 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1447091407008086 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1203451310567055 }
        flops { key: "f32xf32->f32" value: 650580118301965 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1905486821650399 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1897070360424028 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1939912961156278 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1341417492748248 }
        flops { key: "f32xf32->f32" value: 710117355598726 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2349544472647702 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2358090616155377 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2326398210365957 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1413879180223646 }
        flops { key: "f32xf32->f32" value: 756206139665911 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2670999562189054 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2592258501141100 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2671259129501856 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 160164353221957 }
        flops { key: "f32xf32->f32" value: 113924861962864 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 162049777241171 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 173379916680122 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 164457317200183 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 269445878042659 }
        flops { key: "f32xf32->f32" value: 174521222917513 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 271661435547122 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 284887721942159 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 240533562724014 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 406643372088619 }
        flops { key: "f32xf32->f32" value: 276666277763463 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 464421204152249 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 464320788756756 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 466844271304347 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 533668898608349 }
        flops { key: "f32xf32->f32" value: 362199974363299 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 627048294912037 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 642959176047904 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 594541430786268 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 794004214262605 }
        flops { key: "f32xf32->f32" value: 462695103258820 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 904584518955349 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 897027421888053 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 906063455724909 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 284887721942159 }
        flops { key: "f32xf32->f32" value: 194465602463098 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 313501262481751 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 290514562770562 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 292413350762527 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 462819751724137 }
        flops { key: "f32xf32->f32" value: 281065852758327 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 510212318365407 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 498024964749536 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 531423817866864 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 636715928544955 }
        flops { key: "f32xf32->f32" value: 406412499621498 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 804753100243582 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 870131137763371 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 861751062600321 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 802497626307922 }
        flops { key: "f32xf32->f32" value: 503867585171281 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1114996701973001 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1074816640640640 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1078054040160642 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 923251783319002 }
        flops { key: "f32xf32->f32" value: 614093122104661 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1533916891428571 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1524998374151169 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1518394023951566 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 482689064508878 }
        flops { key: "f32xf32->f32" value: 302292180180180 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 544493825557809 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 569926658174097 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 534598866816031 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 699962075619296 }
        flops { key: "f32xf32->f32" value: 420415749412686 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 911495606112054 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 909757952976064 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 920580279927124 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 912900217014719 }
        flops { key: "f32xf32->f32" value: 519594398257924 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1365866527587851 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1372742244027167 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1320715650676506 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1062814759751306 }
        flops { key: "f32xf32->f32" value: 615666619505814 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1898747699381078 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1764571608874281 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1764118620321404 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1159626674586567 }
        flops { key: "f32xf32->f32" value: 734056963937788 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2398083359017309 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2382948773701366 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2380802270509978 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 729258391374480 }
        flops { key: "f32xf32->f32" value: 418449658612626 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 880116249180327 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 880116249180327 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 884283981058266 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 964456811542132 }
        flops { key: "f32xf32->f32" value: 547425968964088 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1462614437595777 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1399582010916497 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1472518143824462 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1158688148917515 }
        flops { key: "f32xf32->f32" value: 626271113444152 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1898747699381078 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1887068231985940 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1893307161560502 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1276364723922734 }
        flops { key: "f32xf32->f32" value: 699734000651678 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2416466584710598 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2347618090188576 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2400177316056023 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1345554305943628 }
        flops { key: "f32xf32->f32" value: 816456096568767 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2895097286289048 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2900045439567859 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2894426616797237 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 982155796021038 }
        flops { key: "f32xf32->f32" value: 489845722627737 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1441143292005704 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1362292378399809 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1427848170212766 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1181884231150247 }
        flops { key: "f32xf32->f32" value: 647027311840916 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1903481157165808 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1898747699381078 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1882106615249781 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1328554407655872 }
        flops { key: "f32xf32->f32" value: 706641542612701 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2472635173287277 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2441275950690966 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2451028167635624 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1411135503223951 }
        flops { key: "f32xf32->f32" value: 755624084447572 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2788939802597402 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2776712800210113 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2781714569948186 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1466234461380899 }
        flops { key: "f32xf32->f32" value: 867100852167768 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 3131369836002825 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 3128198233137212 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 3128767935165898 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 265777679207920 }
        flops { key: "f32xf32->f32" value: 174720010414124 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 289808859379217 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 288562704649287 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 237921964103700 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 402451958020989 }
        flops { key: "f32xf32->f32" value: 266768155031055 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 463619094991364 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 457300606473594 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 478494573975044 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 536736727818045 }
        flops { key: "f32xf32->f32" value: 353204547368421 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 670198532573925 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 644502895558223 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 639132038095238 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 790642421832574 }
        flops { key: "f32xf32->f32" value: 460833400858369 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 840666920336660 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 910577685058567 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 909950698305084 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 791826754730025 }
        flops { key: "f32xf32->f32" value: 511199130657303 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1012754984761399 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1034184275463520 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1035212508451086 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 455602768218945 }
        flops { key: "f32xf32->f32" value: 280496819226750 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 504459395818651 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 530373832551247 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 531423817866864 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 632263697335492 }
        flops { key: "f32xf32->f32" value: 392162828341855 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 798840750674230 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 793967519364081 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 844136654088050 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 801187762160145 }
        flops { key: "f32xf32->f32" value: 499879806331471 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1046340774955843 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1085684351870576 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1112685827979274 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 913239909844779 }
        flops { key: "f32xf32->f32" value: 614093122104661 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1534876189046725 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1524118983676366 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1526285464108031 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 1090646850177755 }
        flops { key: "f32xf32->f32" value: 668548937493311 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1712208215672106 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1717557529017745 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1724410346942360 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 683041872773536 }
        flops { key: "f32xf32->f32" value: 422068327044025 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 936743139803707 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 897777444816053 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 872783437512700 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 903775537061392 }
        flops { key: "f32xf32->f32" value: 524144039539921 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1338830204488778 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1412818189473684 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1401751728459530 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1060485752098765 }
        flops { key: "f32xf32->f32" value: 613205403387289 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1868697360526459 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1755914675388389 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1767476253497942 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1163061297046627 }
        flops { key: "f32xf32->f32" value: 732374979868060 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2400764279485746 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2410194891133558 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2404796918253079 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 1350485933693623 }
        flops { key: "f32xf32->f32" value: 769974753061659 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2651214380246913 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2639807803318992 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2639807803318992 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 945767640187173 }
        flops { key: "f32xf32->f32" value: 535532081795511 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1343857101376721 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1399696039107055 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1396283256176853 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1145324612266666 }
        flops { key: "f32xf32->f32" value: 634177526172019 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1777442365526874 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1777718251655629 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1789196957300562 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1265225848510513 }
        flops { key: "f32xf32->f32" value: 695520143477424 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2422343992950051 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2410110361449163 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2421746431350437 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1345751933573554 }
        flops { key: "f32xf32->f32" value: 814211809668246 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2884888089502739 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2893208013472549 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2884766985118485 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 1539854612058775 }
        flops { key: "f32xf32->f32" value: 844198873934301 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 3122477132679026 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 3118509563260120 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 3129729778020677 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1152431271775951 }
        flops { key: "f32xf32->f32" value: 600023371891589 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1804608107563025 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1862518341717259 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1852876314063848 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1322749398213735 }
        flops { key: "f32xf32->f32" value: 692059948800064 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2274424992917190 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2245147567171981 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2428679156600106 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1411903819196038 }
        flops { key: "f32xf32->f32" value: 783319864536610 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2662049496833175 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2764703763115545 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2767375835051546 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1465710635889068 }
        flops { key: "f32xf32->f32" value: 869531089085859 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 3133333032521344 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 3124606771972900 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 3128661100229916 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 1670467342710337 }
        flops { key: "f32xf32->f32" value: 869747066286761 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 3279676742061148 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 3280107240848911 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 3281752003581683 }
      }
    }
  }
  entries {
    key: "sm_90"
    value {
      entries {
        b: 1
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 6721641025641 }
        flops { key: "f32xf32->f32" value: 6026298850574 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 5991862857142 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 5991862857142 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 6009031518624 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 12372578171091 }
        flops { key: "f32xf32->f32" value: 10155699757869 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 10381940594059 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 10433592039800 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 10356306172839 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 23692449779346 }
        flops { key: "f32xf32->f32" value: 17697485232067 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 15709003745318 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 15887515151515 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 16039403441682 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 43804741514360 }
        flops { key: "f32xf32->f32" value: 33893365656565 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 30229218018018 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 30012908765652 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 30229218018018 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 82646384236453 }
        flops { key: "f32xf32->f32" value: 55924053333333 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 59592730824730 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59493673758865 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59283448763250 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 11459846994535 }
        flops { key: "f32xf32->f32" value: 9058971922246 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 9279433628318 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 9279433628318 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 9279433628318 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 23045626373626 }
        flops { key: "f32xf32->f32" value: 16644063492063 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 16448250980392 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 16448250980392 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 16416062622309 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 43804741514360 }
        flops { key: "f32xf32->f32" value: 26630501587301 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 47662545454545 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 46995002801120 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 47798336182336 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 83055524752475 }
        flops { key: "f32xf32->f32" value: 51781530864197 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 58457198606271 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 58457198606271 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 58457198606271 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 146206675381263 }
        flops { key: "f32xf32->f32" value: 81740394640682 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 89240510638297 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 166937472636815 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 89597949265687 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 18893261261261 }
        flops { key: "f32xf32->f32" value: 15252014545454 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 12748644376899 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 12671613293051 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 12729298937784 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 41120627450980 }
        flops { key: "f32xf32->f32" value: 20945338327091 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 46733192200557 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 46345900552486 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 46603377777777 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 75065843400447 }
        flops { key: "f32xf32->f32" value: 68618470347648 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 90443212938005 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 90933420054200 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 90933420054200 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 141579881856540 }
        flops { key: "f32xf32->f32" value: 72865216069489 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 169466828282828 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 172074010256410 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 172960989690721 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 237553500884955 }
        flops { key: "f32xf32->f32" value: 111662003327787 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 267899656686626 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 273355861507128 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 269513510040160 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 31895847908745 }
        flops { key: "f32xf32->f32" value: 41221660933660 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 42799020408163 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 44384169312169 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 44384169312169 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 63913203809523 }
        flops { key: "f32xf32->f32" value: 67243350701402 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 86480494845360 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 87381333333333 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 87609483028720 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 119623643493761 }
        flops { key: "f32xf32->f32" value: 60241350089766 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 162098705314009 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 164482509803921 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 162885592233009 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 222583296849087 }
        flops { key: "f32xf32->f32" value: 99200094604582 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 277309355371900 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 277309355371900 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 275601084188911 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 345921979381443 }
        flops { key: "f32xf32->f32" value: 143625177100053 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 416825242236024 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 431568257234726 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 416825242236024 }
      }
      entries {
        b: 1
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 90933420054200 }
        flops { key: "f32xf32->f32" value: 61342654478976 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 78398205607476 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 78215459207459 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 79891504761904 }
      }
      entries {
        b: 1
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 93077481276005 }
        flops { key: "f32xf32->f32" value: 108240103225806 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 151146090090090 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 151146090090090 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 151146090090090 }
      }
      entries {
        b: 1
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 177068242744063 }
        flops { key: "f32xf32->f32" value: 160355708482676 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 249475330855018 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 250874257943925 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 250874257943925 }
      }
      entries {
        b: 1
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 296286375275938 }
        flops { key: "f32xf32->f32" value: 123589068139963 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 411710822085889 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 410451767584097 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 422068327044025 }
      }
      entries {
        b: 1
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 418449658612626 }
        flops { key: "f32xf32->f32" value: 184175269982847 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 542293850505050 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 561580451882845 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 540077622885884 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 12336188235294 }
        flops { key: "f32xf32->f32" value: 10230009756097 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 10754625641025 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 10837994832041 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 10699755102040 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 23629881690140 }
        flops { key: "f32xf32->f32" value: 17623126050420 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 16039403441682 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 16194223938223 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 16008793893129 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 45714485013623 }
        flops { key: "f32xf32->f32" value: 34379540983606 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 30393507246376 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 30338546112115 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 30338546112115 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 84519979848866 }
        flops { key: "f32xf32->f32" value: 56205078726968 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 58970882249560 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59388375221238 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59178892416225 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 141281818947368 }
        flops { key: "f32xf32->f32" value: 80659692307692 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 171196081632653 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 170760468193384 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 171196081632653 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 23045626373626 }
        flops { key: "f32xf32->f32" value: 15768060150375 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 16808732373199 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 16878486921529 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 16644063492063 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 43577184415584 }
        flops { key: "f32xf32->f32" value: 26800664536741 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 47934902857142 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 47259763380281 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 48072252148997 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 85380234096692 }
        flops { key: "f32xf32->f32" value: 52510848200312 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 58052650519031 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 58254222222222 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 58146963283873 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 147492008791208 }
        flops { key: "f32xf32->f32" value: 81840078048780 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 177039047650453 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 175677654450261 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 176602273684210 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 243589343012704 }
        flops { key: "f32xf32->f32" value: 133417224652087 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 312134251162790 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 307134389016018 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 309257437788018 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 41527762376237 }
        flops { key: "f32xf32->f32" value: 23899168091168 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 46345900552486 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 46345900552486 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 46733192200557 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 73423264770240 }
        flops { key: "f32xf32->f32" value: 42632487254824 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 91678775956284 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 91428970027247 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 90443212938005 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 146525903930131 }
        flops { key: "f32xf32->f32" value: 72707328277356 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 178481021276595 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 178007596816976 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 178481021276595 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 241833744144144 }
        flops { key: "f32xf32->f32" value: 117631663453111 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 321095043062200 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 310689185185185 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 318051488151658 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 381300363636363 }
        flops { key: "f32xf32->f32" value: 192980198418404 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 505528165725047 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 493447529411764 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 499879806331471 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 61680941176470 }
        flops { key: "f32xf32->f32" value: 67786731313131 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 87154368831168 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 87838827225130 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 88069375328083 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 116306523396880 }
        flops { key: "f32xf32->f32" value: 57456219178082 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 171633923273657 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 169039959697733 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 169466828282828 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 231011580034423 }
        flops { key: "f32xf32->f32" value: 96838187590187 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 322638769230769 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 319566019047619 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 322638769230769 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 342829445721583 }
        flops { key: "f32xf32->f32" value: 170111188846641 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 502688119850187 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 497102696296296 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 502688119850187 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 529981156959526 }
        flops { key: "f32xf32->f32" value: 245370617915904 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 766958445714285 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 766958445714285 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 769156034383954 }
      }
      entries {
        b: 1
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 92820005532503 }
        flops { key: "f32xf32->f32" value: 110365075958474 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 153919412844036 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 151487277652370 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 153216584474885 }
      }
      entries {
        b: 1
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 187454927374301 }
        flops { key: "f32xf32->f32" value: 158275622641509 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 300263373601789 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 291777669565217 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 289887101511879 }
      }
      entries {
        b: 1
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 300936609865470 }
        flops { key: "f32xf32->f32" value: 123532193281178 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 497102696296296 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 497102696296296 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 504577924812030 }
      }
      entries {
        b: 1
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 460413495846063 }
        flops { key: "f32xf32->f32" value: 185064085487762 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 746691115438108 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 748773935843793 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 749819709497206 }
      }
      entries {
        b: 1
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 661578449784350 }
        flops { key: "f32xf32->f32" value: 280570113404755 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 961272895255147 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 975242346957311 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 965595165467625 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 23563505617977 }
        flops { key: "f32xf32->f32" value: 17586180293501 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 16578276679841 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 16644063492063 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 16416062622309 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 45466710027100 }
        flops { key: "f32xf32->f32" value: 33554432000000 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 30448667876588 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 30559591985428 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 30504029090909 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 80854053012048 }
        flops { key: "f32xf32->f32" value: 56394003361344 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 59178892416225 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 58661594405594 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59178892416225 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 144320137634408 }
        flops { key: "f32xf32->f32" value: 83781353308364 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 172516359897172 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 172074010256410 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 172960989690721 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 219668949263502 }
        flops { key: "f32xf32->f32" value: 108065803542673 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 303660018099547 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 305040290909090 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 300936609865470 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 43690666666666 }
        flops { key: "f32xf32->f32" value: 26672839427662 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 47259763380281 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 47798336182336 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 47798336182336 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 84307618090452 }
        flops { key: "f32xf32->f32" value: 46668194714881 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 59283448763250 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59283448763250 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59074704225352 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 149130808888888 }
        flops { key: "f32xf32->f32" value: 81840078048780 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 177536677248677 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 177536677248677 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 178007596816976 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 244032232727272 }
        flops { key: "f32xf32->f32" value: 100087791200596 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 328914634400367 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 324197410628019 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 328965019607843 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 347714321243523 }
        flops { key: "f32xf32->f32" value: 153298614983759 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 532610031746031 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 559240533333333 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 535799313373253 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 69759733887733 }
        flops { key: "f32xf32->f32" value: 41425224691358 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 91428970027247 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 91678775956284 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 91180521739130 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 144010437768240 }
        flops { key: "f32xf32->f32" value: 72865216069489 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 178956970666666 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 178481021276595 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 179435465240641 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 252764082862523 }
        flops { key: "f32xf32->f32" value: 116609668114682 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 331401797530864 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 330585536945812 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 331401797530864 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 384027834048640 }
        flops { key: "f32xf32->f32" value: 195795372720641 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 578524689655172 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 574808256959314 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 577280550537634 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 494356272559852 }
        flops { key: "f32xf32->f32" value: 240641376961004 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 865920825806451 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 834946986003110 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 867319728594507 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 116105301038062 }
        flops { key: "f32xf32->f32" value: 58304834057341 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 168192641604010 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 169895858227848 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 168192641604010 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 235469698245614 }
        flops { key: "f32xf32->f32" value: 99273467455621 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 329722654383540 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 328160704156479 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 330585536945812 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 359833050938337 }
        flops { key: "f32xf32->f32" value: 169681072060682 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 583555339130434 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 582289492407809 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 581029125541125 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 529458493096646 }
        flops { key: "f32xf32->f32" value: 249128033410672 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 887389937190082 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 896278651085141 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 877240052287581 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 639132038095238 }
        flops { key: "f32xf32->f32" value: 283010496573537 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1201053494407158 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1207808575928009 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1209168720720720 }
      }
      entries {
        b: 1
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 187193483960948 }
        flops { key: "f32xf32->f32" value: 161903170084439 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 292413350762527 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 289223386936026 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 291144746203904 }
      }
      entries {
        b: 1
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 312134251162790 }
        flops { key: "f32xf32->f32" value: 122071603456116 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 556920033195020 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 559240533333333 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 572357049040511 }
      }
      entries {
        b: 1
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 472597633802816 }
        flops { key: "f32xf32->f32" value: 184175269982847 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 824686500768049 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 834946986003110 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 825955249230769 }
      }
      entries {
        b: 1
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 665679990080595 }
        flops { key: "f32xf32->f32" value: 283758410147991 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1254371289719626 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1252907612602100 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1257308927400468 }
      }
      entries {
        b: 1
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 751920044817927 }
        flops { key: "f32xf32->f32" value: 325820611136398 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1526285464108031 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1525201454545454 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1546064541396688 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 45343827027027 }
        flops { key: "f32xf32->f32" value: 32961131630648 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 30448667876588 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 30611866347360 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 30504029090909 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 84307618090452 }
        flops { key: "f32xf32->f32" value: 55924053333333 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 59074704225352 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59074704225352 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59074704225352 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 145572373101952 }
        flops { key: "f32xf32->f32" value: 83572682440846 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 173857160621761 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 172516359897172 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 173857160621761 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 219668949263502 }
        flops { key: "f32xf32->f32" value: 114912438356164 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 306433168949771 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 304348589569161 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 300936609865470 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 303316899435028 }
        flops { key: "f32xf32->f32" value: 147492008791208 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 493447529411764 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 503631249530956 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 495268369003690 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 78951604705882 }
        flops { key: "f32xf32->f32" value: 50382030030030 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 58052650519031 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 58559218150087 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 58457198606271 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 150131686800894 }
        flops { key: "f32xf32->f32" value: 82342164417177 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 178007596816976 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 174308737662337 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 178956970666666 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 244922861313868 }
        flops { key: "f32xf32->f32" value: 133682996015936 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 324982392251816 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 330585536945812 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 329773287469287 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 348617475324675 }
        flops { key: "f32xf32->f32" value: 150891206295671 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 548947762781186 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 548947762781186 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 546711723014256 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 449264361506276 }
        flops { key: "f32xf32->f32" value: 195367871906841 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 848137301737756 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 844136654088050 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 848137301737756 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 142784817021276 }
        flops { key: "f32xf32->f32" value: 68688704196519 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 180886425876010 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 179916525469168 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 181375308108108 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 257615600767754 }
        flops { key: "f32xf32->f32" value: 116205825108225 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 327360312195121 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 335544320000000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 325771184466019 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 386794605187319 }
        flops { key: "f32xf32->f32" value: 199877480268056 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 595200567627494 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 587386118161925 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 589968035164835 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 498024964749536 }
        flops { key: "f32xf32->f32" value: 256999000478697 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 940229267950963 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 943534115992970 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 941878792982456 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 596523235555555 }
        flops { key: "f32xf32->f32" value: 302632983089064 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1380130879177378 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1390857284974093 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1367823979617834 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 231409875862068 }
        flops { key: "f32xf32->f32" value: 100688468117029 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 335544320000000 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 330585536945812 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 335544320000000 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 354136485488126 }
        flops { key: "f32xf32->f32" value: 192151364352183 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 589968035164835 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 610080581818181 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 593883752212389 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 535265116650049 }
        flops { key: "f32xf32->f32" value: 266834449304174 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 999759612662942 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1001624835820895 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1005376239700374 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 641039894925373 }
        flops { key: "f32xf32->f32" value: 342610664964901 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1412818189473684 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1409110005249343 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1414679610013175 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 725501232432432 }
        flops { key: "f32xf32->f32" value: 315250095126247 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1835456109401709 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1838599013698630 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1848092640275387 }
      }
      entries {
        b: 1
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 333874945273631 }
        flops { key: "f32xf32->f32" value: 122182729176149 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 550072655737704 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 553475167010309 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 553475167010309 }
      }
      entries {
        b: 1
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 458080982935153 }
        flops { key: "f32xf32->f32" value: 174872958449542 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 917728054700854 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 936947490401396 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 935315177700348 }
      }
      entries {
        b: 1
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 663622882571075 }
        flops { key: "f32xf32->f32" value: 326266127013065 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1416545941952506 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1414679610013175 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1412818189473684 }
      }
      entries {
        b: 1
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 752183414360770 }
        flops { key: "f32xf32->f32" value: 372051914067914 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1856079211754537 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1838599013698630 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1840174505569837 }
      }
      entries {
        b: 1
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 752315168330705 }
        flops { key: "f32xf32->f32" value: 383410756650598 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2112625330054107 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2101256015655577 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2049125618320610 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 84519979848866 }
        flops { key: "f32xf32->f32" value: 55188210526315 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 58457198606271 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 58153261698440 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 58457198606271 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 141879205073995 }
        flops { key: "f32xf32->f32" value: 83886080000000 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 171196081632653 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 173407917312661 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 173407917312661 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 222953036544850 }
        flops { key: "f32xf32->f32" value: 115010906598114 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 304348589569161 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 307838825688073 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 303660018099547 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 308192257175660 }
        flops { key: "f32xf32->f32" value: 132757396636993 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 500812417910447 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 507439425330812 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 484540534296028 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 398567863400148 }
        flops { key: "f32xf32->f32" value: 187651489688919 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 779203065312046 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 739491614325068 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 784898994152046 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 144943550755939 }
        flops { key: "f32xf32->f32" value: 81740394640682 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 175218966057441 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 176138750656167 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 175218966057441 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 240103270125223 }
        flops { key: "f32xf32->f32" value: 130689121713729 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 325771184466019 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 323416212048192 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 325771184466019 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 354604301188903 }
        flops { key: "f32xf32->f32" value: 146206675381263 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 551202168377823 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 562757769392033 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 555766989648033 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 449640629815745 }
        flops { key: "f32xf32->f32" value: 196076938346001 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 854889987261146 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 868723158576051 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 858993459200000 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 551768665981500 }
        flops { key: "f32xf32->f32" value: 237448435205661 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1332185885856079 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1333840775155279 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1333840775155279 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 229040491467576 }
        flops { key: "f32xf32->f32" value: 114422615515771 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 332222099009901 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 328914634400367 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 332222099009901 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 377546351617440 }
        flops { key: "f32xf32->f32" value: 194659503988397 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 597851795100222 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 595200567627494 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 597851795100222 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 497563403151065 }
        flops { key: "f32xf32->f32" value: 274053553854007 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 977906943533697 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1011056331450094 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 977906943533697 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 604584360360360 }
        flops { key: "f32xf32->f32" value: 303660018099547 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1497547871687587 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1359166865822784 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1516584497175141 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 685221329929802 }
        flops { key: "f32xf32->f32" value: 309034918405526 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1880458535901926 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1920826161001789 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1939912961156278 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 314327231850117 }
        flops { key: "f32xf32->f32" value: 191193344729344 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 582289492407809 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 563940033613445 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 577280550537634 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 531029586547972 }
        flops { key: "f32xf32->f32" value: 271283937342091 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1012963984905660 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 997901323420074 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1005376239700374 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 643730110311750 }
        flops { key: "f32xf32->f32" value: 345921979381443 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1574401501466275 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1547178420749279 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1562942975254730 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 727467360433604 }
        flops { key: "f32xf32->f32" value: 310824091474887 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1999519225325884 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2003249671641791 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2018311699248120 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 735565558486042 }
        flops { key: "f32xf32->f32" value: 326514162688155 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2301697371918542 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2266473507124010 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2279706632696390 }
      }
      entries {
        b: 1
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 398863976225854 }
        flops { key: "f32xf32->f32" value: 233625288076588 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 943534115992970 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 932067555555555 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 941878792982456 }
      }
      entries {
        b: 1
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 661986327990135 }
        flops { key: "f32xf32->f32" value: 325567457863518 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1553895548480463 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1538312068767908 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1556147571014492 }
      }
      entries {
        b: 1
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 751656859642982 }
        flops { key: "f32xf32->f32" value: 301398569907281 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2031678001892147 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2025927969811320 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2027721355444083 }
      }
      entries {
        b: 1
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 759364797736916 }
        flops { key: "f32xf32->f32" value: 381673091264551 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2315346251212938 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2321603943783784 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2339306806100218 }
      }
      entries {
        b: 1
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 780193877565849 }
        flops { key: "f32xf32->f32" value: 403149640735319 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2423753698474561 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2443098575654152 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2435479045080805 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 12336188235294 }
        flops { key: "f32xf32->f32" value: 10230009756097 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 10381940594059 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 10381940594059 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 10356306172839 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 24036126074498 }
        flops { key: "f32xf32->f32" value: 17439933471933 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 15917662239089 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 16039403441682 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 15797755178907 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 44267060686015 }
        flops { key: "f32xf32->f32" value: 33354306163021 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 30012908765652 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 30012908765652 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 30066695340501 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 80854053012048 }
        flops { key: "f32xf32->f32" value: 52841625196850 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 59493673758865 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59599346358792 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59918628571428 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 135573462626262 }
        flops { key: "f32xf32->f32" value: 81840078048780 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 84947929113924 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 156796411214953 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 156796411214953 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 22795130434782 }
        flops { key: "f32xf32->f32" value: 16416062622309 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 16352062378167 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 16163021194605 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 16384000000000 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 44034687664041 }
        flops { key: "f32xf32->f32" value: 26504290679304 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 23269370319001 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 43018502564102 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 44384169312169 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 83886080000000 }
        flops { key: "f32xf32->f32" value: 51622203076923 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 59178892416225 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59178892416225 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59178892416225 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 145257281385281 }
        flops { key: "f32xf32->f32" value: 77492914549653 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 157163615925058 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 159783009523809 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 87381333333333 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 236298816901408 }
        flops { key: "f32xf32->f32" value: 112977885521885 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 263689053045186 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 260111875968992 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 265777679207920 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 38130036363636 }
        flops { key: "f32xf32->f32" value: 24244531791907 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 43804741514360 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 44034687664041 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 22733355013550 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 76087147392290 }
        flops { key: "f32xf32->f32" value: 38881149478563 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 84947929113924 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 85163532994923 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 84733414141414 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 145888834782608 }
        flops { key: "f32xf32->f32" value: 108590394822006 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 161319384615384 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 160932527577937 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 162098705314009 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 239247286987522 }
        flops { key: "f32xf32->f32" value: 117838215978928 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 263689053045186 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 266834449304174 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 265252426877470 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 367216766073871 }
        flops { key: "f32xf32->f32" value: 145572373101952 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 404270265060240 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 409825123664122 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 406105077155824 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 61455003663003 }
        flops { key: "f32xf32->f32" value: 69042041152263 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 82040176039119 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 81840078048780 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 82443321867321 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 125203104477611 }
        flops { key: "f32xf32->f32" value: 112034831385642 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 155344592592592 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 157163615925058 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 155705020881670 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 220390357963875 }
        flops { key: "f32xf32->f32" value: 165089456334563 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 269513510040160 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 262657001956947 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 267365992031872 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 343267846547314 }
        flops { key: "f32xf32->f32" value: 154628718894009 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 392449497076023 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 409200390243902 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 389036892753623 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 484103617673579 }
        flops { key: "f32xf32->f32" value: 168139966176010 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 547269023445463 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 542293850505050 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 546711723014256 }
      }
      entries {
        b: 2
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 92820005532503 }
        flops { key: "f32xf32->f32" value: 107031681020733 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 147492008791208 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 144631172413793 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 146846529540481 }
      }
      entries {
        b: 2
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 190379756028368 }
        flops { key: "f32xf32->f32" value: 159973454112038 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 247178136279926 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 246723764705882 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 247178136279926 }
      }
      entries {
        b: 2
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 298925897550111 }
        flops { key: "f32xf32->f32" value: 202440012066365 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 407337566009104 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 390735743813682 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 392449497076023 }
      }
      entries {
        b: 2
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 421405739403453 }
        flops { key: "f32xf32->f32" value: 183734056125941 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 535799313373253 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 546711723014256 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 521740439261418 }
      }
      entries {
        b: 2
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 602887043234138 }
        flops { key: "f32xf32->f32" value: 174308737662337 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 628286614394382 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 626819511967308 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 635726361160450 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 23497501400560 }
        flops { key: "f32xf32->f32" value: 17476266666666 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 16288559223300 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 16480565815324 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 16448250980392 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 45100043010752 }
        flops { key: "f32xf32->f32" value: 33825032258064 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 30504029090909 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 30559591985428 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 30393507246376 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 81442796116504 }
        flops { key: "f32xf32->f32" value: 53008581358609 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 60349697841726 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 60025817531305 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 60025817531305 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 141879205073995 }
        flops { key: "f32xf32->f32" value: 82342164417177 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 161708106024096 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 160932527577937 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 160547521531100 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 214405316293929 }
        flops { key: "f32xf32->f32" value: 113647525825571 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 280790225941422 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 281378884696016 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 276168164609053 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 43240247422680 }
        flops { key: "f32xf32->f32" value: 26379270440251 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 24070611190817 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 24036126074498 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 44384169312169 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 84733414141414 }
        flops { key: "f32xf32->f32" value: 51701744221879 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 61794534069981 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 62137837037037 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 61567765137614 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 147816881057268 }
        flops { key: "f32xf32->f32" value: 78398205607476 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 166523235732009 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 165700898765432 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 166111049504950 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 244476735883424 }
        flops { key: "f32xf32->f32" value: 112977885521885 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 300936609865470 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 297600283813747 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 296286375275938 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 345921979381443 }
        flops { key: "f32xf32->f32" value: 184238473575840 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 460438174957118 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 481930800718132 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 454975349152542 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 69184395876288 }
        flops { key: "f32xf32->f32" value: 36711632385120 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 86037005128205 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 85816961636828 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 85163532994923 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 145888834782608 }
        flops { key: "f32xf32->f32" value: 72160068817204 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 167772160000000 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 167772160000000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 168615236180904 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 250406208955223 }
        flops { key: "f32xf32->f32" value: 117625220353836 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 300936609865470 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 301612871910112 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 305735143507972 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 381300363636363 }
        flops { key: "f32xf32->f32" value: 146525903930131 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 484540534296028 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 489845722627737 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 491640029304029 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 495268369003690 }
        flops { key: "f32xf32->f32" value: 215345949810724 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 698141628088426 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 682173966963151 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 693631669250646 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 114520245733788 }
        flops { key: "f32xf32->f32" value: 107374182400000 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 159403477434679 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 158649796690307 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 158649796690307 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 235469698245614 }
        flops { key: "f32xf32->f32" value: 98762125091979 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 305040290909090 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 307134389016018 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 307134389016018 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 371280022130013 }
        flops { key: "f32xf32->f32" value: 154361964347326 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 492542121100917 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 501748515887850 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 489845722627737 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 530996760338752 }
        flops { key: "f32xf32->f32" value: 182175402782490 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 743588520775623 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 731431760217983 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 731431760217983 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 641039894925373 }
        flops { key: "f32xf32->f32" value: 232559516792332 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 972592231884058 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 932016990397656 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 956989147950089 }
      }
      entries {
        b: 2
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 174308737662337 }
        flops { key: "f32xf32->f32" value: 155524597914252 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 276168164609053 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 276168164609053 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 275601084188911 }
      }
      entries {
        b: 2
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 322638769230769 }
        flops { key: "f32xf32->f32" value: 121025904418394 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 487178686025408 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 482797582733812 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 488064465454545 }
      }
      entries {
        b: 2
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 474686924845269 }
        flops { key: "f32xf32->f32" value: 184745668272539 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 690953554697554 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 684784326530612 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 686535693094629 }
      }
      entries {
        b: 2
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 662803595061728 }
        flops { key: "f32xf32->f32" value: 190819588413008 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 947697991173874 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 939406670166229 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 933688542608695 }
      }
      entries {
        b: 2
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 747731075208913 }
        flops { key: "f32xf32->f32" value: 252911063603642 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1045512973709834 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1038403649792982 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1052172292013718 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 45221606469002 }
        flops { key: "f32xf32->f32" value: 33288126984126 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 30559591985428 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 30504029090909 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 30559591985428 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 86258179948586 }
        flops { key: "f32xf32->f32" value: 54207483037156 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 59599346358792 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59493673758865 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59811821746880 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 142784817021276 }
        flops { key: "f32xf32->f32" value: 81840078048780 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 161319384615384 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 162098705314009 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 162098705314009 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 217180789644012 }
        flops { key: "f32xf32->f32" value: 114033753610875 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 286790017094017 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 287404128479657 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 284963329087048 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 301612871910112 }
        flops { key: "f32xf32->f32" value: 157440150146627 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 457300606473594 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 470939396491228 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 461229305841924 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 81640953771289 }
        flops { key: "f32xf32->f32" value: 51622203076923 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 57456219178082 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 57358003418803 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 57653663230240 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 150131686800894 }
        flops { key: "f32xf32->f32" value: 78306725787631 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 166937472636815 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 166111049504950 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 167353775561097 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 246723764705882 }
        flops { key: "f32xf32->f32" value: 112316090376569 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 303660018099547 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 303660018099547 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 300936609865470 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 348617475324675 }
        flops { key: "f32xf32->f32" value: 196935544775092 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 515231201535508 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 512281404580152 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 519217516441005 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 456135014443500 }
        flops { key: "f32xf32->f32" value: 234032655623365 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 749819709497206 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 751920044817927 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 742560044260027 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 140985008403361 }
        flops { key: "f32xf32->f32" value: 67108864000000 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 170760468193384 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 168615236180904 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 170327065989847 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 244922861313868 }
        flops { key: "f32xf32->f32" value: 113551377326565 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 309971658198614 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 312861836829836 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 310689185185185 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 382386689458689 }
        flops { key: "f32xf32->f32" value: 139737353461738 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 545600520325203 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 546711723014256 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 553475167010309 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 499385767804197 }
        flops { key: "f32xf32->f32" value: 214405316293929 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 817155117199391 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 858993459200000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 867319728594507 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 597851795100222 }
        flops { key: "f32xf32->f32" value: 274053553854007 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1179936070329670 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1154561101075268 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1173488332240437 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 233016888888888 }
        flops { key: "f32xf32->f32" value: 92119236787920 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 314327231850117 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 309971658198614 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 315065089201877 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 355073354497354 }
        flops { key: "f32xf32->f32" value: 151572815358554 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 556920033195020 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 553475167010309 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 547827461224489 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 532082172447968 }
        flops { key: "f32xf32->f32" value: 169734717673095 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 858993459200000 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 860370051282051 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 863136514469453 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 642190086124401 }
        flops { key: "f32xf32->f32" value: 236819987648875 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1202398459126539 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1199711535195530 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1206451487640449 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 721600688172043 }
        flops { key: "f32xf32->f32" value: 289496312752763 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1460873229931972 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1473907788606726 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1487177041551246 }
      }
      entries {
        b: 2
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 326563815085158 }
        flops { key: "f32xf32->f32" value: 110421824763471 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 516222030769230 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 540111581488933 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 529458493096646 }
      }
      entries {
        b: 2
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 452673618887015 }
        flops { key: "f32xf32->f32" value: 183294951177876 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 848137301737756 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 819650247328244 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 819650247328244 }
      }
      entries {
        b: 2
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 662394709438618 }
        flops { key: "f32xf32->f32" value: 180642971736204 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1185145501103752 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1201053494407158 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1207808575928009 }
      }
      entries {
        b: 2
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 754827292794376 }
        flops { key: "f32xf32->f32" value: 300305362606628 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1481023205517241 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1478983228650137 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1490273176960444 }
      }
      entries {
        b: 2
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 742688448210271 }
        flops { key: "f32xf32->f32" value: 313592822429906 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1603796600448095 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1614042576475009 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1631206720850740 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 84733414141414 }
        flops { key: "f32xf32->f32" value: 52841625196850 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 60133390681003 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59918628571428 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 60787014492753 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 144923987582669 }
        flops { key: "f32xf32->f32" value: 82342164417177 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 163281907542579 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 160932527577937 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 161708106024096 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 222214781456953 }
        flops { key: "f32xf32->f32" value: 112504382229673 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 286790017094017 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 289262344827586 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 287404128479657 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 304348589569161 }
        flops { key: "f32xf32->f32" value: 174876518566775 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 464421204152249 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 475107001769911 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 462819751724137 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 396800378418329 }
        flops { key: "f32xf32->f32" value: 216131607085346 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 752974631136044 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 758292248587570 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 757222724964739 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 144320137634408 }
        flops { key: "f32xf32->f32" value: 64839482125603 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 165700898765432 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 164482509803921 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 165292768472906 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 242270267148014 }
        flops { key: "f32xf32->f32" value: 112693306465155 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 305735143507972 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 304348589569161 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 305040290909090 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 348165312581063 }
        flops { key: "f32xf32->f32" value: 186924633154894 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 527378106090373 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 536870912000000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 526344031372549 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 450395060402684 }
        flops { key: "f32xf32->f32" value: 240965400359066 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 810983250755287 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 823421644171779 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 818400780487804 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 547827461224489 }
        flops { key: "f32xf32->f32" value: 285266159404888 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1195703590200445 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1207808575928009 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1203746439461883 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 229432013675213 }
        flops { key: "f32xf32->f32" value: 88534121372031 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 312134251162790 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 312134251162790 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 309971658198614 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 382386689458689 }
        flops { key: "f32xf32->f32" value: 140615744368779 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 565127275789473 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 556847827823155 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 560408050104384 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 502688119850187 }
        flops { key: "f32xf32->f32" value: 211117149823043 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 956989147950089 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 956989147950089 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 956989147950089 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 599186285714285 }
        flops { key: "f32xf32->f32" value: 278315662001036 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1348921889447236 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1352319677581864 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1337162919053549 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 683911989808917 }
        flops { key: "f32xf32->f32" value: 309703439284684 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1712506896331738 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1726273028938906 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1662139046439628 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 319566019047619 }
        flops { key: "f32xf32->f32" value: 110421824763471 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 558077871101871 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 541200516129032 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 553475167010309 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 529458493096646 }
        flops { key: "f32xf32->f32" value: 168721216844751 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 950214003539823 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 948535180212014 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 955286320284697 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 639132038095238 }
        flops { key: "f32xf32->f32" value: 228115960059485 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1385473321290322 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1383687917525773 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1387263338501292 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 723774321572262 }
        flops { key: "f32xf32->f32" value: 301610224348452 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1806125860386879 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1786592053244592 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1794054843776107 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 722807492516277 }
        flops { key: "f32xf32->f32" value: 321720396704119 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2014524998123827 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2041334266159695 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2008871513564078 }
      }
      entries {
        b: 2
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 398863976225854 }
        flops { key: "f32xf32->f32" value: 242160988723500 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 884466082372322 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 890333187396351 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 896278651085141 }
      }
      entries {
        b: 2
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 663597249179188 }
        flops { key: "f32xf32->f32" value: 179645612180023 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1394469901298701 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1362508460940598 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1399924151238592 }
      }
      entries {
        b: 2
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 751393858642407 }
        flops { key: "f32xf32->f32" value: 301655239218991 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1794054843776107 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1764571608874281 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1773314325350949 }
      }
      entries {
        b: 2
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 752183414360770 }
        flops { key: "f32xf32->f32" value: 316503132981148 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2054025488283118 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2022112662900188 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2069863757108433 }
      }
      entries {
        b: 2
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 771777749855402 }
        flops { key: "f32xf32->f32" value: 328536451406402 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2095105998048780 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2093063984405458 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2096128499755978 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 143089262260127 }
        flops { key: "f32xf32->f32" value: 77852510440835 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 161319384615384 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 162491196125908 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 161708106024096 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 218595648208469 }
        flops { key: "f32xf32->f32" value: 113743837288135 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 286178524520255 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 286178524520255 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 286178524520255 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 308192257175660 }
        flops { key: "f32xf32->f32" value: 176023249836065 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 474267590106007 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 483667488288288 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 470114633975481 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 400052840536512 }
        flops { key: "f32xf32->f32" value: 216480206451612 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 709208602377807 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 702710617801047 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 700875864229765 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 467453993905093 }
        flops { key: "f32xf32->f32" value: 229285036087977 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 955286320284697 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 916161965870307 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 944363961301671 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 238821580071174 }
        flops { key: "f32xf32->f32" value: 112977885521885 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 307838825688073 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 306433168949771 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 306433168949771 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 348617475324675 }
        flops { key: "f32xf32->f32" value: 191876666190135 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 525314003913894 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 528416251968503 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 560408050104384 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 453055621940928 }
        flops { key: "f32xf32->f32" value: 239140718040089 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 827227907550077 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 836247526479750 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 827227907550077 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 548667258048032 }
        flops { key: "f32xf32->f32" value: 284434920264900 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1269198373522458 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1267699910271546 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1261741273795534 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 613216347230154 }
        flops { key: "f32xf32->f32" value: 212243886934176 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1569797988304093 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1581357620029455 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1573248093772893 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 378078107042253 }
        flops { key: "f32xf32->f32" value: 146605929000546 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 566319527426160 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 550072655737704 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 558077871101871 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 495725680517082 }
        flops { key: "f32xf32->f32" value: 210620208709297 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 922458611683848 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 969081068592057 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 974357372050816 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 598518296544035 }
        flops { key: "f32xf32->f32" value: 259604835275095 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1418417204755614 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1439332203753351 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1445143773889636 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 684347880178457 }
        flops { key: "f32xf32->f32" value: 279109202453190 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1893620191127032 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1887068231985940 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1922545790510295 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 691843958762886 }
        flops { key: "f32xf32->f32" value: 313547035771645 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2218474842975206 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2218474842975206 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2221917897568546 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 524800500488758 }
        flops { key: "f32xf32->f32" value: 161316355086480 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 956989147950089 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 958698057142857 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 970833475587703 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 636480037937166 }
        flops { key: "f32xf32->f32" value: 212622143366336 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1489239700416088 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1487177041551246 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1508064359550561 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 721843243025210 }
        flops { key: "f32xf32->f32" value: 284887721942159 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1973676740076971 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1970168484403669 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1979247601843318 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 730560860010205 }
        flops { key: "f32xf32->f32" value: 275264378130895 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2288208468833244 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2284557072340425 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2215042442496132 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 759163463720724 }
        flops { key: "f32xf32->f32" value: 326649966778608 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2395366670826289 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2367025238908790 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2378165723145072 }
      }
      entries {
        b: 2
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 597519100723427 }
        flops { key: "f32xf32->f32" value: 171059713876055 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1427848170212766 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1464859241473397 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1398101333333333 }
      }
      entries {
        b: 2
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 741022652864044 }
        flops { key: "f32xf32->f32" value: 230317851565851 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1959382890510949 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1943424115837104 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1984735349353050 }
      }
      entries {
        b: 2
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 746561323831044 }
        flops { key: "f32xf32->f32" value: 300200412105962 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2312852609585352 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2264083972588297 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2346976664480874 }
      }
      entries {
        b: 2
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 778986660500019 }
        flops { key: "f32xf32->f32" value: 319078587332873 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2368289653679803 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2378824312378842 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2397414064192018 }
      }
      entries {
        b: 2
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 804185691260323 }
        flops { key: "f32xf32->f32" value: 336088741093054 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2537646851403249 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2555767507289497 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2535025702228124 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 23899168091168 }
        flops { key: "f32xf32->f32" value: 17225067761806 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 16070130268199 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 16100975047984 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 16100975047984 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 45590260869565 }
        flops { key: "f32xf32->f32" value: 32961131630648 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 30840470588235 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 30783882568807 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 30727501831501 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 80082176610978 }
        flops { key: "f32xf32->f32" value: 52758540880503 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 60458436036036 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 60567566787003 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 60458436036036 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 135573462626262 }
        flops { key: "f32xf32->f32" value: 82040176039119 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 155344592592592 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 155705020881670 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 155705020881670 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 213382715421303 }
        flops { key: "f32xf32->f32" value: 106437532117367 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 249475330855018 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 249475330855018 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 247634184501845 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 43233283298437 }
        flops { key: "f32xf32->f32" value: 26337858712715 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 23172950276243 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 44501899204244 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 23205001383125 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 84519979848866 }
        flops { key: "f32xf32->f32" value: 51701744221879 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 58355533913043 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 58559218150087 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 58867424561403 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 141879205073995 }
        flops { key: "f32xf32->f32" value: 77852510440835 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 88301136842105 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 87953950196592 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 159783009523809 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 234236872600349 }
        flops { key: "f32xf32->f32" value: 112598765100671 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 260616947572815 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 260111875968992 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 262144000000000 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 346368330322580 }
        flops { key: "f32xf32->f32" value: 185640011065006 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 375434204195804 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 376487315568022 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 382386689458689 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 73746004395604 }
        flops { key: "f32xf32->f32" value: 38926255220417 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 85163532994923 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 84307618090452 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 84519979848866 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 144320137634408 }
        flops { key: "f32xf32->f32" value: 59283448763250 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 163281907542579 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 159025744075829 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 161708106024096 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 241398791366906 }
        flops { key: "f32xf32->f32" value: 116914397212543 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 265777679207920 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 269513510040160 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 265252426877470 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 365218307482993 }
        flops { key: "f32xf32->f32" value: 147411013728720 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 408577558599695 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 400649934328358 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 401248813153961 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 491640029304029 }
        flops { key: "f32xf32->f32" value: 189506146134839 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 531029586547972 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 534199912437810 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 528416251968503 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 122685308957952 }
        flops { key: "f32xf32->f32" value: 106861248407643 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 151830009049773 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 153919412844036 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 153567194508009 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 231409875862068 }
        flops { key: "f32xf32->f32" value: 157903209411764 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 267365992031872 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 268973402805611 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 270600258064516 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 356015193633952 }
        flops { key: "f32xf32->f32" value: 154096128587830 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 397682157037037 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 419430400000000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 395922501474926 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 494356272559852 }
        flops { key: "f32xf32->f32" value: 166937472636815 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 547269023445463 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 539568755778894 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 548387039836567 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 601536035854341 }
        flops { key: "f32xf32->f32" value: 201793238864875 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 611818703133903 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 602887043234138 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 581973888346883 }
      }
      entries {
        b: 4
        m: 256
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 186932768802228 }
        flops { key: "f32xf32->f32" value: 154450780207134 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 241833744144144 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 256140702290076 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 235883528998242 }
      }
      entries {
        b: 4
        m: 256
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 303316899435028 }
        flops { key: "f32xf32->f32" value: 202899059712774 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 401248813153961 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 397093869822485 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 395922501474926 }
      }
      entries {
        b: 4
        m: 256
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 422733001574803 }
        flops { key: "f32xf32->f32" value: 180699972484590 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 536334577422577 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 545600520325203 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 534199912437810 }
      }
      entries {
        b: 4
        m: 256
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 589644054914881 }
        flops { key: "f32xf32->f32" value: 172571813564770 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 621378370370370 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 604584360360360 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 615324827507163 }
      }
      entries {
        b: 4
        m: 256
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 660560949861581 }
        flops { key: "f32xf32->f32" value: 226122317363377 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 643730110311750 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 649768123449319 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 643718050246360 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 44739242666666 }
        flops { key: "f32xf32->f32" value: 33091155818540 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 30393507246376 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 30559591985428 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 30559591985428 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 83886080000000 }
        flops { key: "f32xf32->f32" value: 53687091200000 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 59074704225352 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59493673758865 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59493673758865 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 135847902834008 }
        flops { key: "f32xf32->f32" value: 81840078048780 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 160932527577937 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 160164353221957 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 162098705314009 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 211699886435331 }
        flops { key: "f32xf32->f32" value: 112788006722689 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 277883494824016 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 276168164609053 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 277883494824016 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 299593142857142 }
        flops { key: "f32xf32->f32" value: 164785424186617 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 417473493001555 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 424068650868878 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 418123763239875 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 80082176610978 }
        flops { key: "f32xf32->f32" value: 51228140458015 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 57752895008605 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 57358003418803 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 58153261698440 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 146846529540481 }
        flops { key: "f32xf32->f32" value: 78306725787631 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 162491196125908 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 162491196125908 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 163281907542579 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 242270267148014 }
        flops { key: "f32xf32->f32" value: 111662003327787 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 298261617777777 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 296286375275938 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 298925897550111 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 345032719794344 }
        flops { key: "f32xf32->f32" value: 184872903581267 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 455747803056027 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 479349028571428 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 481876730169415 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 448888722408026 }
        flops { key: "f32xf32->f32" value: 231509664510564 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 629391456037514 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 626453806301050 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 630130178403755 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 138941747412008 }
        flops { key: "f32xf32->f32" value: 71697504273504 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 167772160000000 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 164886643734643 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 166937472636815 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 247178136279926 }
        flops { key: "f32xf32->f32" value: 117220723144104 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 307134389016018 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 302974555304740 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 302292180180180 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 382932176890156 }
        flops { key: "f32xf32->f32" value: 145809590439978 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 484540534296028 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 485416737793851 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 481930800718132 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 494356272559852 }
        flops { key: "f32xf32->f32" value: 210207874706343 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 690065439588689 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 693631669250646 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 691843958762886 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 598184860167130 }
        flops { key: "f32xf32->f32" value: 255409568030447 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 932877344917463 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 928792192463642 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 939406670166229 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 224444361204013 }
        flops { key: "f32xf32->f32" value: 97826332361516 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 308546501149425 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 305735143507972 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 309971658198614 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 358391797062750 }
        flops { key: "f32xf32->f32" value: 153831206876790 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 496183837338262 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 490741235831809 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 494356272559852 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 529458493096646 }
        flops { key: "f32xf32->f32" value: 176544199934232 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 734433532147742 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 738474431911967 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 735439605479452 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 637992765300059 }
        flops { key: "f32xf32->f32" value: 228212927523910 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 951055645704163 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 951055645704163 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 956136975957257 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 702710617801047 }
        flops { key: "f32xf32->f32" value: 277919457486734 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1006790270979840 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1018700179904533 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1013442023596035 }
      }
      entries {
        b: 4
        m: 512
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 317675095857988 }
        flops { key: "f32xf32->f32" value: 121574028985507 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 482797582733812 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 478494573975044 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 474267590106007 }
      }
      entries {
        b: 4
        m: 512
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 454590103302286 }
        flops { key: "f32xf32->f32" value: 184809264027538 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 692736660645161 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 687414740076824 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 671928550688360 }
      }
      entries {
        b: 4
        m: 512
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 665267548946716 }
        flops { key: "f32xf32->f32" value: 185895118690284 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 932877344917463 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 941827157721616 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 956989147950089 }
      }
      entries {
        b: 4
        m: 512
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 747991517938000 }
        flops { key: "f32xf32->f32" value: 250991543712014 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1027996001914791 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1061009707509881 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1027996001914791 }
      }
      entries {
        b: 4
        m: 512
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 724155672905075 }
        flops { key: "f32xf32->f32" value: 305125619895478 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1118772413649387 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1118772413649387 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1120816100208768 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 83886080000000 }
        flops { key: "f32xf32->f32" value: 52593153605015 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 59918628571428 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 59178892416225 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 59599346358792 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 141879205073995 }
        flops { key: "f32xf32->f32" value: 82342164417177 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 160547521531100 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 160547521531100 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 160932527577937 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 216829932148626 }
        flops { key: "f32xf32->f32" value: 112977885521885 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 284359593220339 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 284359593220339 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 279038935550935 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 302292180180180 }
        flops { key: "f32xf32->f32" value: 172398639104082 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 466844271304347 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 472597633802816 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 461229305841924 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 398863976225854 }
        flops { key: "f32xf32->f32" value: 205934373609512 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 652334036452004 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 683041872773536 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 646832424096385 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 142784817021276 }
        flops { key: "f32xf32->f32" value: 60622280036133 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 164080352078239 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 163680156097560 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 164482509803921 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 243589343012704 }
        flops { key: "f32xf32->f32" value: 111941391159299 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 304348589569161 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 300936609865470 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 304348589569161 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 347264496765847 }
        flops { key: "f32xf32->f32" value: 195652664723032 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 548947762781186 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 515231201535508 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 519154755953100 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 451911542087542 }
        flops { key: "f32xf32->f32" value: 237448435205661 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 743588520775623 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 741534408839779 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 735439605479452 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 549228554475703 }
        flops { key: "f32xf32->f32" value: 274614277237851 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1047552999024390 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1049601000977517 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1061009707509881 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 225955771043771 }
        flops { key: "f32xf32->f32" value: 83209998760074 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 309971658198614 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 312134251162790 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 312861836829836 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 379146124293785 }
        flops { key: "f32xf32->f32" value: 137942166495375 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 547827461224489 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 543391611336032 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 551202168377823 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 499414801860465 }
        flops { key: "f32xf32->f32" value: 209306398440545 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 863136514469453 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 861751062600321 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 867319728594507 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 601199229563269 }
        flops { key: "f32xf32->f32" value: 269581175997991 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1147083473592842 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1150848685959271 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1141064637619553 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 679798559037670 }
        flops { key: "f32xf32->f32" value: 303445478027412 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1382796940115904 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1383687917525773 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1393564988968202 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 323027022864019 }
        flops { key: "f32xf32->f32" value: 202287457422758 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 553475167010309 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 528416251968503 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 554618710743801 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 526344031372549 }
        flops { key: "f32xf32->f32" value: 165902516406898 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 853530861685214 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 857621265175718 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 861751062600321 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 642190086124401 }
        flops { key: "f32xf32->f32" value: 223789458941225 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1214640072398190 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1203746439461883 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1214640072398190 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 721116067159167 }
        flops { key: "f32xf32->f32" value: 285038976373772 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1467863053998633 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1406341616240995 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1472896877914952 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 714391657771355 }
        flops { key: "f32xf32->f32" value: 318546858710969 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1580775596613912 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1603759171415902 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1607397940119760 }
      }
      entries {
        b: 4
        m: 1024
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 403359062359128 }
        flops { key: "f32xf32->f32" value: 242379644243792 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 825955249230769 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 786048187408492 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 825955249230769 }
      }
      entries {
        b: 4
        m: 1024
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 661986327990135 }
        flops { key: "f32xf32->f32" value: 175821487473391 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1186455054143646 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1185145501103752 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1198372571428571 }
      }
      entries {
        b: 4
        m: 1024
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 749296457780879 }
        flops { key: "f32xf32->f32" value: 245818255993475 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1464859241473397 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1466860415300546 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1462863520435967 }
      }
      entries {
        b: 4
        m: 1024
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 740639299189515 }
        flops { key: "f32xf32->f32" value: 310440758471456 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1599615380260707 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1602599737313432 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1607397940119760 }
      }
      entries {
        b: 4
        m: 1024
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 769776377094721 }
        flops { key: "f32xf32->f32" value: 325561288307750 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1657328688404399 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1663104470861568 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1661496052611218 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 143395008547008 }
        flops { key: "f32xf32->f32" value: 78398205607476 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 160932527577937 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 160932527577937 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 160932527577937 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 220029062295081 }
        flops { key: "f32xf32->f32" value: 112882866274179 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 285569634042553 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 286790017094017 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 286178524520255 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 303660018099547 }
        flops { key: "f32xf32->f32" value: 172516359897172 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 466844271304347 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 484540534296028 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 479349028571428 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 400649934328358 }
        flops { key: "f32xf32->f32" value: 209388031201248 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 702710617801047 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 693631669250646 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 702710617801047 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 460228486806504 }
        flops { key: "f32xf32->f32" value: 222122843194042 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 887389937190082 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 890333187396351 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 896278651085141 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 230614652920962 }
        flops { key: "f32xf32->f32" value: 113168404721753 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 305040290909090 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 305735143507972 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 305735143507972 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 345921979381443 }
        flops { key: "f32xf32->f32" value: 188640517217146 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 524288000000000 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 527378106090373 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 520160747971418 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 451152026890756 }
        flops { key: "f32xf32->f32" value: 240533562724014 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 823421644171779 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 824686500768049 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 806112480480480 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 547548099949005 }
        flops { key: "f32xf32->f32" value: 280570113404755 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1173488332240437 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1176058952902519 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1170928924754634 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 608697179138322 }
        flops { key: "f32xf32->f32" value: 296490908187215 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1459880114208021 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1438368150033489 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1468867064295485 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 378611362482369 }
        flops { key: "f32xf32->f32" value: 119195384675158 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 553475167010309 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 555766989648033 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 563940033613445 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 493447529411764 }
        flops { key: "f32xf32->f32" value: 242598694984184 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 958698057142857 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 950108902997456 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 962134250896057 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 597851795100222 }
        flops { key: "f32xf32->f32" value: 261760561677230 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1320715650676506 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1332185885856079 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1335499781094527 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 685440040855410 }
        flops { key: "f32xf32->f32" value: 309569503820095 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1707061723370429 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1677721600000000 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1660853556071152 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 688296041025641 }
        flops { key: "f32xf32->f32" value: 329672036843721 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1912273951914514 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1920826161001789 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1919109605004468 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 523744563868056 }
        flops { key: "f32xf32->f32" value: 156567778361038 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 932067555555555 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 936947490401396 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 940229267950963 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 635350191715976 }
        flops { key: "f32xf32->f32" value: 283758410147991 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1374829480153649 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1355734626262626 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1387263338501292 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 722328842246888 }
        flops { key: "f32xf32->f32" value: 301105390914189 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1758790866502866 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1774779874380165 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1779191091963546 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 723180214850985 }
        flops { key: "f32xf32->f32" value: 320496029848518 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2012636970946579 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1932059062528115 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1944303891353553 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 751328137146855 }
        flops { key: "f32xf32->f32" value: 334316750681092 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2067870628791526 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2076336674149834 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2064391875030040 }
      }
      entries {
        b: 4
        m: 2048
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 573886597541421 }
        flops { key: "f32xf32->f32" value: 166419997520148 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1330535097893432 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1335499781094527 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1333840775155279 }
      }
      entries {
        b: 4
        m: 2048
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 743331134648667 }
        flops { key: "f32xf32->f32" value: 304823796735273 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1731841651612903 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1711142349003984 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1738853156275303 }
      }
      entries {
        b: 4
        m: 2048
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 740639299189515 }
        flops { key: "f32xf32->f32" value: 314325794104059 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2021161080470588 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2012636970946579 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2035529524170616 }
      }
      entries {
        b: 4
        m: 2048
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 775330457067102 }
        flops { key: "f32xf32->f32" value: 328297881650387 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2059936352997602 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2053534447047573 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2050073141391089 }
      }
      entries {
        b: 4
        m: 2048
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 800699997215239 }
        flops { key: "f32xf32->f32" value: 335092321690103 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2201418398769861 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2221038186052149 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2215328070148291 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 256
        flops { key: "bf16xbf16->bf16" value: 210702869701726 }
        flops { key: "f32xf32->f32" value: 112128427736006 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 285569634042553 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 285569634042553 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 286178524520255 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 256
        flops { key: "bf16xbf16->bf16" value: 304003913929784 }
        flops { key: "f32xf32->f32" value: 173857160621761 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 466033777777777 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 477643160142348 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 474267590106007 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 256
        flops { key: "bf16xbf16->bf16" value: 396507320531757 }
        flops { key: "f32xf32->f32" value: 206966427139552 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 728454426051560 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 709208602377807 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 728454426051560 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 256
        flops { key: "bf16xbf16->bf16" value: 463419000431592 }
        flops { key: "f32xf32->f32" value: 219354815934627 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 941878792982456 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 949373849690539 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 941878792982456 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 256
        flops { key: "bf16xbf16->bf16" value: 505171406257351 }
        flops { key: "f32xf32->f32" value: 241724859072489 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1053204339382050 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1061009707509881 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1052655812260653 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 512
        flops { key: "bf16xbf16->bf16" value: 341520936386768 }
        flops { key: "f32xf32->f32" value: 169359909148264 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 519217516441005 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 522247968871595 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 519217516441005 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 512
        flops { key: "bf16xbf16->bf16" value: 451152026890756 }
        flops { key: "f32xf32->f32" value: 235057316987740 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 834946986003110 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 834946986003110 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 829785026275115 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 512
        flops { key: "bf16xbf16->bf16" value: 547548099949005 }
        flops { key: "f32xf32->f32" value: 275955236186070 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1267699910271546 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1263225675294117 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1264713573616018 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 512
        flops { key: "bf16xbf16->bf16" value: 612680557907312 }
        flops { key: "f32xf32->f32" value: 289845530119363 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1596642117472119 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1602599737313432 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1594271453600594 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 512
        flops { key: "bf16xbf16->bf16" value: 620391058211757 }
        flops { key: "f32xf32->f32" value: 316736541295440 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1777718251655629 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1761676495488105 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1771120534432989 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 489399190519598 }
        flops { key: "f32xf32->f32" value: 163129965474675 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 880116249180327 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 875808991843393 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 960413080500894 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 596192017767906 }
        flops { key: "f32xf32->f32" value: 230463065047957 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1424060774535809 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1439332203753351 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1435483721925133 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 677653407384032 }
        flops { key: "f32xf32->f32" value: 278640670559231 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1869002304612706 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1821444994062765 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1892056077533039 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 690176329101719 }
        flops { key: "f32xf32->f32" value: 295713908970028 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2198038534288638 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2210482396294390 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2189076093781855 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 1024
        flops { key: "bf16xbf16->bf16" value: 722328842246888 }
        flops { key: "f32xf32->f32" value: 322142681117569 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2260509103157894 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2261104130560674 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2277289128313892 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 576970351423965 }
        flops { key: "f32xf32->f32" value: 153347875464153 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1295225360675512 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1425951957503320 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1405421235602094 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 710146708994709 }
        flops { key: "f32xf32->f32" value: 243976783458304 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1898747699381078 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1885411455662862 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1830761848252344 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 716544427093760 }
        flops { key: "f32xf32->f32" value: 299696362080620 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2243974553814002 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2276082297827239 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2256945504992117 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 758158392939099 }
        flops { key: "f32xf32->f32" value: 292981216299122 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2362468259625962 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2363118182118294 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2372254789284728 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 2048
        flops { key: "bf16xbf16->bf16" value: 759530558969895 }
        flops { key: "f32xf32->f32" value: 340836184539995 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2527938373160683 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2536148388544434 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2547406579342940 }
      }
      entries {
        b: 4
        m: 4096
        n: 256
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 611296227725590 }
        flops { key: "f32xf32->f32" value: 166008321583178 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 1681662997650744 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 1697615532015810 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 1672495052959501 }
      }
      entries {
        b: 4
        m: 4096
        n: 512
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 720624539759440 }
        flops { key: "f32xf32->f32" value: 264794531196054 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2204808673511293 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2203677422267829 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2192428430832057 }
      }
      entries {
        b: 4
        m: 4096
        n: 1024
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 773519549031967 }
        flops { key: "f32xf32->f32" value: 310430240689167 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2360520635339379 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2388747105672970 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2350830484948002 }
      }
      entries {
        b: 4
        m: 4096
        n: 2048
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 804715405124361 }
        flops { key: "f32xf32->f32" value: 335819386905045 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2501072817586257 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2495260593173565 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2554627387955390 }
      }
      entries {
        b: 4
        m: 4096
        n: 4096
        k: 4096
        flops { key: "bf16xbf16->bf16" value: 788699393418751 }
        flops { key: "f32xf32->f32" value: 345601440030376 }
        flops { key: "f8e4m3fnxf8e4m3fn->bf16" value: 2592796435858738 }
        flops { key: "f8e4m3fnxf8e5m2->bf16" value: 2595734559794515 }
        flops { key: "f8e5m2xf8e4m3fn->bf16" value: 2620467860643587 }
      }
    }
  }
)pb";
// END_DEFAULT_PERF_TABLE

#endif  // XLA_BACKENDS_GPU_COST_MODEL_MATMUL_INTERPOLATOR_DATA_H_
