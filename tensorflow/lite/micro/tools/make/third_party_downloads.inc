# Add URLs and MD5 checksums for third-party libraries here.
# We use mirror.tensorflow.org to cache copies of third-party files,
# but this is just an optimization applied manually by TensorFlow
# engineers, so add non-mirrored URLs if you need to update this
# in a pull request and we'll periodically copy them and update
# the URL.

GEMMLOWP_URL := "https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip"
GEMMLOWP_MD5 := "7e8191b24853d75de2af87622ad293ba"

LEON_BCC2_URL := "http://mirror.tensorflow.org/www.gaisler.com/anonftp/bcc2/bin/bcc-2.0.7-gcc-linux64.tar.xz"
LEON_BCC2_MD5 := "cdf78082be4882da2a92c9baa82fe765"

TSIM_URL := "http://mirror.tensorflow.org/www.gaisler.com/anonftp/tsim/tsim-eval-2.0.63.tar.gz"
TSIM_MD5 := "afa0095d3ed989a949e1467f94e41d2f"

AM_SDK_URL := "http://mirror.tensorflow.org/s3.asia.ambiqmicro.com/downloads/AmbiqSuite-Rel2.2.0.zip"
AM_SDK_MD5 := "7605fa2d4d97e6bb7a1190c92b66b597"
AM_SDK_DEST := AmbiqSuite-Rel2.2.0

SF_BSPS_URL := "http://mirror.tensorflow.org/github.com/sparkfun/SparkFun_Apollo3_AmbiqSuite_BSPs/archive/v0.0.7.zip"
SF_BSPS_MD5 := "34199f7e754735661d1c8a70a40ca7a3"
SF_BSPS_DEST := boards_sfe

ifeq ($(HOST_OS),osx)
  RISCV_TOOLCHAIN_URL := "http://mirror.tensorflow.org/static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-8.1.0-2019.01.0-x86_64-apple-darwin.tar.gz"
  RISCV_TOOLCHAIN_MD5 := "2ac2fa00618b9ab7fa0c7d0ec173de94"
else
  RISCV_TOOLCHAIN_URL := "http://mirror.tensorflow.org/static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-20181030-x86_64-linux-ubuntu14.tar.gz"
  RISCV_TOOLCHAIN_MD5="2366b7afe36a54dc94fb0ff8a0830934"
endif

SIFIVE_FE310_LIB_URL := "http://mirror.tensorflow.org/github.com/sifive/freedom-e-sdk/archive/baeeb8fd497a99b3c141d7494309ec2e64f19bdf.zip"
SIFIVE_FE310_LIB_MD5 := "06ee24c4956f8e21670ab3395861fe64"

KISSFFT_URL="http://mirror.tensorflow.org/github.com/mborgerding/kissfft/archive/v130.zip"
KISSFFT_MD5="438ba1fef5783cc5f5f201395cc477ca"

RUY_URL="https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip"
RUY_MD5="abf7a91eb90d195f016ebe0be885bb6e"

IMAGE_RECOGNITION_MODEL_URL := "https://storage.googleapis.com/download.tensorflow.org/models/tflite/cifar_image_recognition_model_2020_05_27.zip"
IMAGE_RECOGNITION_MODEL_MD5 := "1f4607b05ac45b8a6146fb883dbc2d7b"

PERSON_MODEL_URL := "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_grayscale_2020_05_27.zip"
PERSON_MODEL_MD5 := "55b85f76e2995153e660391d4a209ef1"

PERSON_MODEL_INT8_URL := "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_int8_grayscale_2020_12_1.zip"
PERSON_MODEL_INT8_MD5 := "e765cc76889db8640cfe876a37e4ec00"

EMBARC_MLI_URL := "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/archive/ef7dd3c4e37d74a908f30713a7d0121387d3c678.zip"
EMBARC_MLI_MD5 := "65c4ff3f4a2963e90fd014f97c69f451"

EMBARC_MLI_PRE_COMPILED_URL := "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/releases/download/Release_1.1/embARC_MLI_package.zip"
EMBARC_MLI_PRE_COMPILED_MD5 := "173990c2dde4efef6a2c95b92d1f0244"

ZEPHYR_URL := "http://mirror.tensorflow.org/github.com/antmicro/zephyr/archive/55e36b9.zip"
ZEPHYR_MD5 := "755622eb4812fde918a6382b65d50c3b"

ETHOSU_URL := "https://git.mlplatform.org/ml/ethos-u/ethos-u-core-driver.git/snapshot/ethos-u-core-driver-dfed5fd24699246811300a05ebc421a978e52149.tar.gz"
ETHOSU_MD5 := "b7ded51a4186daa711c2517c4ab23901"

HIMAX_WE1_SDK_URL ="https://www.himax.com.tw/we-i/himax_we1_sdk_v04.zip"
HIMAX_WE1_SDK_MD5 ="40b3ccb3c2e41210fe5c970d61e7e7d3"
