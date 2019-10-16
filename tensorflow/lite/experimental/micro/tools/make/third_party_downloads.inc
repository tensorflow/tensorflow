# Add URLs and MD5 checksums for third-party libraries here.

GEMMLOWP_URL := "https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip"
GEMMLOWP_MD5 := "7e8191b24853d75de2af87622ad293ba"

FLATBUFFERS_URL := "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v1.11.0.tar.gz"
FLATBUFFERS_MD5 := "02c64880acb89dbd57eebacfd67200d8"

ifeq ($(HOST_OS),osx)
  GCC_EMBEDDED_URL := "https://developer.arm.com/-/media/Files/downloads/gnu-rm/7-2018q2/gcc-arm-none-eabi-7-2018-q2-update-mac.tar.bz2"
  GCC_EMBEDDED_MD5 := "a66be9828cf3c57d7d21178e07cd8904"
else
  GCC_EMBEDDED_URL := "https://developer.arm.com/-/media/Files/downloads/gnu-rm/7-2018q2/gcc-arm-none-eabi-7-2018-q2-update-linux.tar.bz2"
  GCC_EMBEDDED_MD5 := "299ebd3f1c2c90930d28ab82e5d8d6c0"
endif

LEON_BCC2_URL := "https://www.gaisler.com/anonftp/bcc2/bin/bcc-2.0.7-gcc-linux64.tar.xz"
LEON_BCC2_MD5 := "cdf78082be4882da2a92c9baa82fe765"

TSIM_URL := "https://www.gaisler.com/anonftp/tsim/tsim-eval-2.0.63.tar.gz"
TSIM_MD5 := "afa0095d3ed989a949e1467f94e41d2f"

CMSIS_URL := "https://github.com/ARM-software/CMSIS_5/archive/01c7adb7685da540be9297b5a93e6640ea3333ce.zip"
CMSIS_MD5 := "3dec53cc74f1d5d79036952137be5d5e"

AM_SDK_URL := "http://s3.asia.ambiqmicro.com/downloads/AmbiqSuite-Rel2.0.0.zip"
AM_SDK_MD5 := "70332bc6968602bd85bee600ca81d06f"

AP3_URL := "https://github.com/AmbiqMicro/TFLiteMicro_Apollo3/archive/dfbcef9a57276c087d95aab7cb234f1d4c9eaaba.zip"
AP3_MD5 := "fc9cbda4562ea97ce21b6df542b66597"

CUST_CMSIS_URL := "https://github.com/AmbiqMicro/TFLiteMicro_CustCMSIS/archive/8f63966c5692e6a3a83956efd2e4aed77c4c9949.zip"
CUST_CMSIS_MD5 := "4fb327201034ee0a820b72de1e807d27"

SPARKFUN_EDGE_BSP_URL := "https://github.com/sparkfun/SparkFun_Edge_BSP/archive/620f5f7a69fc69e38cda8132b69302d9c28ba0dd.zip"
SPARKFUN_EDGE_BSP_MD5 := "10fb37d721c782327edc981d3b5b07cf"

STM32_BARE_LIB_URL := "https://github.com/google/stm32_bare_lib/archive/c07d611fb0af58450c5a3e0ab4d52b47f99bc82d.zip"
STM32_BARE_LIB_MD5 := "282bff40d4d0b92278fd123a3b6e3123"

ifeq ($(HOST_OS),osx)
  RISCV_TOOLCHAIN_URL := "https://static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-8.1.0-2019.01.0-x86_64-apple-darwin.tar.gz"
  RISCV_TOOLCHAIN_MD5 := "2ac2fa00618b9ab7fa0c7d0ec173de94"
else
  RISCV_TOOLCHAIN_URL := "https://static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-20181030-x86_64-linux-ubuntu14.tar.gz"
  RISCV_TOOLCHAIN_MD5="2366b7afe36a54dc94fb0ff8a0830934"
endif

SIFIVE_FE310_LIB_URL := "https://github.com/sifive/freedom-e-sdk/archive/baeeb8fd497a99b3c141d7494309ec2e64f19bdf.zip"
SIFIVE_FE310_LIB_MD5 := "06ee24c4956f8e21670ab3395861fe64"

KISSFFT_URL="https://github.com/mborgerding/kissfft/archive/v130.zip"
KISSFFT_MD5="438ba1fef5783cc5f5f201395cc477ca"

PERSON_MODEL_URL := "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_grayscale_2019_11_07.zip"
PERSON_MODEL_MD5 := "e6430de25aa92bcb807d07278a1b5b90"
