# TensorFlow Bazel Clang

This is a specialized toolchain that uses an old Debian with a new Clang that
can cross compile to any x86_64 microarchitecture. It's intended to build Linux
binaries that only require the following ABIs:

- GLIBC_2.18
- CXXABI_1.3.7 (GCC 4.8.3)
- GCC_4.2.0

Which are available on at least the following Linux platforms:

- Ubuntu 14+
- CentOS 7+
- Debian 8+
- SuSE 13.2+
- Mint 17.3+
- Manjaro 0.8.11

# System Install

On Debian 8 (Jessie) Clang 6.0 can be installed as follows:

```sh
cat >>/etc/apt/sources.list <<'EOF'
deb http://apt.llvm.org/jessie/ llvm-toolchain-jessie main
deb-src http://apt.llvm.org/jessie/ llvm-toolchain-jessie main
EOF
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
apt-key fingerprint |& grep '6084 F3CF 814B 57C1 CF12  EFD5 15CF 4D18 AF4F 7421'
apt-get update
apt-get install clang lld
```

# Bazel Configuration

This toolchain can compile TensorFlow in 2m30s on a 96-core Skylake GCE VM if
the following `.bazelrc` settings are added:

```
startup --host_jvm_args=-Xmx30G
startup --host_jvm_args=-Xms30G
startup --host_jvm_args=-XX:MaxNewSize=3g
startup --host_jvm_args=-XX:-UseAdaptiveSizePolicy
startup --host_jvm_args=-XX:+UseConcMarkSweepGC
startup --host_jvm_args=-XX:TargetSurvivorRatio=70
startup --host_jvm_args=-XX:SurvivorRatio=6
startup --host_jvm_args=-XX:+UseCMSInitiatingOccupancyOnly
startup --host_jvm_args=-XX:CMSFullGCsBeforeCompaction=1
startup --host_jvm_args=-XX:CMSInitiatingOccupancyFraction=75

build --jobs=100
build --local_resources=200000,100,100
build --crosstool_top=@local_config_clang6//clang6
build --noexperimental_check_output_files
build --nostamp
build --config=opt
build --noexperimental_check_output_files
build --copt=-march=native
build --host_copt=-march=native
```

# x86_64 Microarchitectures

## Intel CPU Line

- 2003 P6 M           SSE SSE2
- 2004 prescott       SSE3 SSSE3 (-march=prescott)
- 2006 core           X64 SSE4.1 (only on 45nm variety) (-march=core2)
- 2008 nehalem        SSE4.2 VT-x VT-d (-march=nehalem)
- 2010 westmere       CLMUL AES (-march=westmere)
- 2012 sandybridge    AVX TXT (-march=sandybridge)
- 2012 ivybridge      F16C MOVBE (-march=ivybridge)
- 2013 haswell        AVX2 TSX BMI2 FMA (-march=haswell)
- 2014 broadwell      RDSEED ADCX PREFETCHW (-march=broadwell - works on trusty gcc4.9)
- 2015 skylake        SGX ADX MPX AVX-512[xeon-only] (-march=skylake / -march=skylake-avx512 - needs gcc7)
- 2018 cannonlake     AVX-512 SHA (-march=cannonlake - needs clang5)

## Intel Low Power CPU Line

- 2013 silvermont     SSE4.1 SSE4.2 VT-x (-march=silvermont)
- 2016 goldmont       SHA (-march=goldmont - needs clang5)

## AMD CPU Line

- 2003 k8             SSE SSE2 (-march=k8)
- 2005 k8 (Venus)     SSE3 (-march=k8-sse3)
- 2008 barcelona      SSE4a?! (-march=barcelona)
- 2011 bulldozer      SSE4.1 SSE4.2 CLMUL AVX AES FMA4?! (-march=bdver1)
- 2011 piledriver     FMA (-march=bdver2)
- 2015 excavator      AVX2 BMI2 MOVBE (-march=bdver4)

## Google Compute Engine Supported CPUs

- 2012 sandybridge 2.6gHz -march=sandybridge
- 2012 ivybridge   2.5gHz -march=ivybridge
- 2013 haswell     2.3gHz -march=haswell
- 2014 broadwell   2.2gHz -march=broadwell
- 2015 skylake     2.0gHz -march=skylake-avx512

See: <https://cloud.google.com/compute/docs/cpu-platforms>
