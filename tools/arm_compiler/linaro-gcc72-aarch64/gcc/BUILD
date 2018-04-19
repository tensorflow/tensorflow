package(default_visibility = ['//tools/arm_compiler/linaro-gcc72-aarch64:__pkg__'])

filegroup(
  name = 'gcc',
  srcs = [
    '@LinaroAarch64Gcc72//:gcc',
    'aarch64-linux-gnu-gcc',
  ],
)

filegroup(
  name = 'ar',
  srcs = [
    '@LinaroAarch64Gcc72//:ar',
    'aarch64-linux-gnu-ar',
  ],
)

filegroup(
  name = 'ld',
  srcs = [
    '@LinaroAarch64Gcc72//:ld',
    'aarch64-linux-gnu-ld',
  ],
)

filegroup(
  name = 'nm',
  srcs = [
    '@LinaroAarch64Gcc72//:nm',
    'aarch64-linux-gnu-nm',
  ],
)

filegroup(
  name = 'objcopy',
  srcs = [
    '@LinaroAarch64Gcc72//:objcopy',
    'aarch64-linux-gnu-objcopy',
  ],
)

filegroup(
  name = 'objdump',
  srcs = [
    '@LinaroAarch64Gcc72//:objdump',
    'aarch64-linux-gnu-objdump',
  ],
)

filegroup(
  name = 'strip',
  srcs = [
    '@LinaroAarch64Gcc72//:strip',
    'aarch64-linux-gnu-strip',
  ],
)

filegroup(
  name = 'as',
  srcs = [
    '@LinaroAarch64Gcc72//:as',
    'aarch64-linux-gnu-as',
  ],
)

filegroup(
  name = 'tool-wrappers',
  srcs = [
    ':gcc',
    ':ar',
    ':ld',
    ':nm',
    ':objcopy',
    ':objdump',
    ':strip',
    ':as',
  ],
)
