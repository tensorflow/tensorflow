/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

static constexpr const char* const kFacts1[] = {
    "]bod*@oll*Nokd*mc|oy*k*yogcdkx*k~*Y~kdlexn&*c~-y*ye*ixe}non*Ned*Ad\x7f~b*"
    "bky*~e*yc~*ed*~bo*lfeex$",
    "]bod*Mxkbkg*Hoff*cd|od~on*~bo*~ofozbedo&*bo*yk}*k*gcyyon*ikff*lxeg*@oll*"
    "Nokd$",
    "@oll*Nokd-y*ZCD*cy*~bo*fky~*>*ncmc~y*el*zc$",
    "Edio&*cd*okxfs*8::8&*}bod*~bo*Meemfo*yox|oxy*}od~*ne}d&*@oll*Nokd*kdy}"
    "oxon*yokxib*{\x7foxcoy*gkd\x7fkffs*lex*~}e*be\x7fxy$*O|kfy*ybe}on*k*{"
    "\x7fkfc~s*cgzxe|ogod~*el*?*zecd~y$",
    "@oll*Nokd*z\x7f~y*bcy*zkd~y*ed*edo*fom*k~*k*~cgo&*h\x7f~*cl*bo*bkn*gexo*~"
    "bkd*~}e*fomy&*se\x7f*}e\x7f\x66n*yoo*~bk~*bcy*kzzxekib*cy*ki~\x7fkffs*"
    "E\"fem*d#$",
    "@oll*Nokd*iegzcfoy*kdn*x\x7f\x64y*bcy*ieno*holexo*y\x7fhgc~~cdm&*h\x7f~*"
    "edfs*~e*iboia*lex*iegzcfox*h\x7fmy$",
    "@oll*Nokd*ixok~on*~bo*}exfn-y*lcxy~*E\";%d#*kfmexc~bg$",
    "@oll*Nokd*}xe~o*kd*E\"dT8#*kfmexc~bg*edio$*C~*}ky*lex*~bo*^xk|ofcdm*"
    "Ykfoygkd*Zxehfog$",
    "^bo*xk~o*k~*}bcib*@oll*Nokd*zxen\x7fioy*ieno*`\x7fgzon*hs*k*lki~ex*el*>:*"
    "cd*fk~o*8:::*}bod*bo*\x7fzmxknon*bcy*aoshekxn*~e*_YH8$:$",
    "@oll*Nokd*ikd*hok~*se\x7f*k~*ieddoi~*le\x7fx$*Cd*~bxoo*ge|oy$",
    "@oll*Nokd*ade}y*}bs*~bo*kdy}ox*cy*>8$",
    "@oll*Nokd*y~kx~y*bcy*zxemxkggcdm*yoyycedy*}c~b*(ik~*4*%no|%gog($",
    "]bod*@oll*Nokd*yksy*(ezod*~bo*zen*hks*neexy(&*Bkf*ezody*~bo*zen*hks*"
    "neexy$",
    "@oll*Nokd*ycgzfs*}kfay*cd~e*Gexnex$",
    "Ib\x7fia*Dexxcy*cy*@oll*Nokd-y*8:/*zxe`oi~$",
    "@oll*Nokd-y*}k~ib*ncyzfksy*yoiedny*ycdio*@kd\x7fkxs*;y~&*;3=:$*Bo*cy*do|"
    "ox*fk~o$",
    "]bod*se\x7fx*ieno*bky*\x7f\x64nolcdon*hobk|cex&*se\x7f*mo~*k*"
    "yomlk\x7f\x66~*kdn*iexx\x7fz~on*nk~k$*]bod*@oll*Nokd-y*ieno*bky*"
    "\x7f\x64nolcdon*hobk|cex&*k*\x7f\x64\x63iexd*xcnoy*cd*ed*k*xkcdhe}*kdn*mc|"
    "oy*o|oxshens*lxoo*cio*ixokg$",
    "Moell*Bcd~ed*neoyd-~*doon*~e*gkao*bcnnod*\x7f\x64\x63~y$*^bos*bcno*hs*~"
    "bogyof|oy*}bod*bo*kzzxekiboy$",
    "Moell*Bcd~ed*neoyd-~*ncykmxoo&*bo*ied~xky~c|ofs*nc|oxmoy$",
    "Nooz*Hofcol*Do~}exay*ki~\x7fkffs*hofco|o*noozfs*cd*Moell*Bcd~ed$",
    "Moell*Bcd~ed*bky*ncyie|oxon*be}*~bo*hxkcd*xokffs*}exay$$$*edio*k*sokx&*"
    "lex*~bo*fky~*8?*sokxy$",
    "Gkxae|*xkdneg*lcofny*~bcda*Moell*Bcd~ed*cy*cd~xki~khfo$",
    "Moell*Bcd~ed*ncnd-~*cd|od~*femci&*h\x7f~*bcy*mxok~'mxok~'mxkdnlk~box*ncn$*"
    "\"^x\x7fo+#",
    "Moell*Bcd~ed*bky*}xc~~od*~}e*zkzoxy*~bk~*kxo*noy~cdon*~e*xo|ef\x7f~cedcpo*"
    "gkibcdo*fokxdcdm$*Dehens*ade}y*}bcib*~}e$"};
static constexpr uint64 kNum1 = sizeof(kFacts1) / sizeof(kFacts1[0]);

static constexpr const char* const kFacts2[] = {
    "Yoxmos*Hxcd*kdn*Hk~gkd*bk|o*do|ox*hood*yood*k~*~bo*ykgo*zfkio*k~*~bo*ykgo*"
    "~cgo$"};
static constexpr uint64 kNum2 = sizeof(kFacts2) / sizeof(kFacts2[0]);

static void E(string* s) {
  for (size_t j = 0; j < s->size(); ++j) {
    (*s)[j] ^= '\n';
  }
}

template <const char* const FACTS[], uint64 N>
class FactOpKernel : public OpKernel {
 public:
  explicit FactOpKernel(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));
    auto output = output_tensor->template scalar<string>();

    string coded = FACTS[context->env()->NowMicros() % N];
    E(&coded);
    output() = coded;
  }
};

REGISTER_KERNEL_BUILDER(Name("Fact").Device(DEVICE_GPU).HostMemory("fact"),
                        FactOpKernel<kFacts1, kNum1>);

static string D(const char* s) {
  string ret(s);
  E(&ret);
  return ret;
}

REGISTER_KERNEL_BUILDER(Name("Fact")
                            .Device(DEVICE_CPU)
                            .Label(D("Yoxmos").c_str()),
                        FactOpKernel<kFacts2, kNum2>);
REGISTER_KERNEL_BUILDER(Name("Fact")
                            .Device(DEVICE_CPU)
                            .Label(D("yoxmos").c_str()),
                        FactOpKernel<kFacts2, kNum2>);

}  // namespace tensorflow
