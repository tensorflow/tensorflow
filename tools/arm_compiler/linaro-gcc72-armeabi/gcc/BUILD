package(default_visibility = ['//tools/arm_compiler/linaro-gcc72-armeabi:__pkg__'])

filegroup(
  name = 'gcc',
  srcs = [
    '@LinaroArmGcc72//:gcc',
    'arm-linux-gnueabihf-gcc',
  ],
)

filegroup(
  name = 'ar',
  srcs = [
    '@LinaroArmGcc72//:ar',
    'arm-linux-gnueabihf-ar',
  ],
)

filegroup(
  name = 'ld',
  srcs = [
    '@LinaroArmGcc72//:ld',
    'arm-linux-gnueabihf-ld',
  ],
)

filegroup(
  name = 'nm',
  srcs = [
    '@LinaroArmGcc72//:nm',
    'arm-linux-gnueabihf-nm',
  ],
)

filegroup(
  name = 'objcopy',
  srcs = [
    '@LinaroArmGcc72//:objcopy',
    'arm-linux-gnueabihf-objcopy',
  ],
)

filegroup(
  name = 'objdump',
  srcs = [
    '@LinaroArmGcc72//:objdump',
    'arm-linux-gnueabihf-objdump',
  ],
)

filegroup(
  name = 'strip',
  srcs = [
    '@LinaroArmGcc72//:strip',
    'arm-linux-gnueabihf-strip',
  ],
)

filegroup(
  name = 'as',
  srcs = [
    '@LinaroArmGcc72//:as',
    'arm-linux-gnueabihf-as',
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
