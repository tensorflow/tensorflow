# Copyright 2026 The OpenXLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Architecture info for TPU platforms."""

import dataclasses

import immutabledict
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

immutabledict = immutabledict.immutabledict


@dataclasses.dataclass(frozen=True)
class PlatformInfo:
  """Information about a TPU platform that is needed for cost modeling.

  Attributes:
    chip_version: The chip version of the TPU platform.
    default_internal_scratch_bytes: The default size of the internal scratch in
      bytes for Pallas kernels.
    matmul_cadence_cycles_by_dtype: A mapping from dtype to the number of cycles
      between each matmul operation for a given MXU.
    latch_cadence_cycles_by_dtype: A mapping from dtype to the number of cycles
      between each latch operation for a given MXU.
    clock_speed_ghz_by_p_state: A mapping from P-state to clock speed in GHz.
    hbm_to_vmem_bandwidth_gb_per_sec: The bandwidth between HBM and VMEM in GB
      per second.
    hbm_to_vmem_latency_ns: The minimum latency of a HBM->VMEM transfer in
      nanoseconds.
    vmem_to_hbm_latency_ns: The minimum latency of a VMEM->HBM transfer in
      nanoseconds.
    matmul_latency_cycles_by_dtype: A mapping from dtype to the number of cycles
      between the start of a matmul and the result being available.
    default_p_state: The default P-state for the TPU platform.
    default_vmem_limit_kib: The default VMEM limit in KiB for the TPU platform.
  """

  chip_version: pltpu.ChipVersion
  default_internal_scratch_bytes: int
  matmul_cadence_cycles_by_dtype: immutabledict[jnp.dtype, int]
  latch_cadence_cycles_by_dtype: immutabledict[jnp.dtype, int]
  clock_speed_ghz_by_p_state: immutabledict[int | None, float]
  hbm_to_vmem_bandwidth_gb_per_sec: int
  hbm_to_vmem_latency_ns: int
  vmem_to_hbm_latency_ns: int
  matmul_latency_cycles_by_dtype: immutabledict[jnp.dtype, int]
  default_p_state: int | None
  default_vmem_limit_kib: int
  _tpu_info: pltpu.TpuInfo = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    tpu_info = pltpu.get_tpu_info_for_chip(
        self.chip_version, num_tensor_cores_per_logical_device=1
    )
    object.__setattr__(self, "_tpu_info", tpu_info)

  @property
  def vreg_size(self) -> tuple[int, int]:
    """Returns the number of sublanes and lanes on the TPU platform.

    Returns:
      A tuple of (sublanes, lanes).
    """
    return self._tpu_info.num_sublanes, self._tpu_info.num_lanes

  @property
  def num_mxus(self) -> int:
    """Returns the number of MXUs on the TPU platform."""
    return self._tpu_info.num_mxus

  @property
  def mxu_size(self) -> tuple[int, int]:
    """Returns the size of the MXU.

    Returns:
      A tuple of (contracting_dim, non_contracting_dim).
    """
    return self._tpu_info.mxu_column_size, self._tpu_info.mxu_column_size

  def mxu_size_by_dtype(self, lhs_dtype: jnp.dtype) -> tuple[int, int]:
    """Returns the MXU size for the given LHS dtype."""
    if (
        jax.dtypes.itemsize_bits(lhs_dtype) < 8
        and lhs_dtype in self.matmul_cadence_cycles_by_dtype
    ):
      # Currently, platforms that support sub-byte dtypes natively use 2x the
      # normal MXU contracting dimension.
      return 2 * self.mxu_size[0], self.mxu_size[1]
    else:
      return self.mxu_size


_PLATFORM_INFOS = (
    PlatformInfo(
        chip_version=pltpu.ChipVersion.TPU_V5E,
        default_internal_scratch_bytes=73728,
        matmul_cadence_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 32,
            jnp.float8_e5m2: 32,
            jnp.bfloat16: 16,
            jnp.float32: 8,
            jnp.uint8: 16,
            jnp.int8: 16,
            jnp.int4: 16,
            jnp.uint4: 16,
        }),
        latch_cadence_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 4,
            jnp.float8_e5m2: 4,
            jnp.bfloat16: 4,
            jnp.float32: 2,
            jnp.uint8: 4,
            jnp.int8: 4,
            jnp.int4: 4,
            jnp.uint4: 4,
        }),
        # Note that V5E doesn't support P states.
        clock_speed_ghz_by_p_state=immutabledict({
            None: 1.5,
        }),
        hbm_to_vmem_bandwidth_gb_per_sec=1459,
        hbm_to_vmem_latency_ns=1133,
        vmem_to_hbm_latency_ns=439,
        matmul_latency_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 131,
            jnp.float8_e5m2: 131,
            jnp.bfloat16: 131,
            jnp.float32: 131,
            jnp.uint8: 121,
            jnp.int8: 121,
            jnp.int4: 121,
            jnp.uint4: 121,
        }),
        default_p_state=None,
        default_vmem_limit_kib=32 * 1024,
    ),
    PlatformInfo(
        chip_version=pltpu.ChipVersion.TPU_V5P,
        default_internal_scratch_bytes=73728,
        matmul_cadence_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 32,
            jnp.float8_e5m2: 32,
            jnp.bfloat16: 16,
            jnp.float32: 8,
            jnp.uint8: 16,
            jnp.int8: 16,
            jnp.int4: 16,
            jnp.uint4: 16,
        }),
        latch_cadence_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 4,
            jnp.float8_e5m2: 4,
            jnp.bfloat16: 4,
            jnp.float32: 2,
            jnp.uint8: 4,
            jnp.int8: 4,
            jnp.int4: 4,
            jnp.uint4: 4,
        }),
        # Note that V5P doesn't support P states.
        clock_speed_ghz_by_p_state=immutabledict({
            None: 1.75,
        }),
        hbm_to_vmem_bandwidth_gb_per_sec=1459,
        hbm_to_vmem_latency_ns=667,
        vmem_to_hbm_latency_ns=624,
        matmul_latency_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 131,
            jnp.float8_e5m2: 131,
            jnp.bfloat16: 131,
            jnp.float32: 131,
            jnp.uint8: 121,
            jnp.int8: 121,
            jnp.int4: 121,
            jnp.uint4: 121,
        }),
        default_p_state=None,
        default_vmem_limit_kib=32 * 1024,
    ),
    PlatformInfo(
        chip_version=pltpu.ChipVersion.TPU_V6E,
        default_internal_scratch_bytes=73728,
        matmul_cadence_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 16,
            jnp.float8_e5m2: 16,
            jnp.bfloat16: 8,
            jnp.float32: 4,
            jnp.uint8: 8,
            jnp.int8: 8,
            jnp.int4: 8,
            jnp.uint4: 8,
        }),
        latch_cadence_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 4,
            jnp.float8_e5m2: 4,
            jnp.bfloat16: 4,
            jnp.float32: 2,
            jnp.uint8: 4,
            jnp.int8: 4,
            jnp.int4: 4,
            jnp.uint4: 4,
        }),
        # Note that GLC doesn't support P states.
        clock_speed_ghz_by_p_state=immutabledict({
            None: 1.75,
        }),
        hbm_to_vmem_bandwidth_gb_per_sec=1373,
        hbm_to_vmem_latency_ns=628,
        vmem_to_hbm_latency_ns=523,
        matmul_latency_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 192,
            jnp.float8_e5m2: 192,
            jnp.bfloat16: 192,
            jnp.float32: 192,
            jnp.uint8: 182,
            jnp.int8: 182,
            jnp.int4: 182,
            jnp.uint4: 182,
        }),
        default_p_state=None,
        default_vmem_limit_kib=32 * 1024,
    ),
    PlatformInfo(
        chip_version=pltpu.ChipVersion.TPU_7X,
        default_internal_scratch_bytes=73728,
        matmul_cadence_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 8,
            jnp.float8_e5m2: 8,
            jnp.bfloat16: 8,
            jnp.float32: 4,
        }),
        latch_cadence_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 4,
            jnp.float8_e5m2: 4,
            jnp.bfloat16: 4,
            jnp.float32: 2,
        }),
        clock_speed_ghz_by_p_state=immutabledict({
            0: 1.6,
            1: 1.7,
            2: 1.8,
            3: 1.9,
            4: 2.0,
            5: 2.05,
            6: 2.1,
            7: 2.2,
        }),
        hbm_to_vmem_bandwidth_gb_per_sec=3207,
        hbm_to_vmem_latency_ns=692,
        vmem_to_hbm_latency_ns=594,
        matmul_latency_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 204,
            jnp.float8_e5m2: 204,
            jnp.bfloat16: 211,
            jnp.float32: 211,
        }),
        default_p_state=3,
        default_vmem_limit_kib=32 * 1024,
    ),
    PlatformInfo(
        chip_version=pltpu.ChipVersion.TPU_8I,
        default_internal_scratch_bytes=73728,
        matmul_cadence_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 4,
            jnp.float8_e5m2: 4,
            jnp.bfloat16: 16,
            jnp.float32: 8,
        }),
        latch_cadence_cycles_by_dtype=immutabledict({
            jnp.int4: 2,
            jnp.float8_e4m3fn: 2,
            jnp.float8_e5m2: 2,
            jnp.bfloat16: 2,
            jnp.float32: 1,
        }),
        clock_speed_ghz_by_p_state=immutabledict({
            0: 1.8,
            1: 1.9,
            2: 2.0,
            3: 2.1,
            4: 2.2,
            5: 2.25,
            6: 2.3,
            7: 2.4,
        }),
        hbm_to_vmem_bandwidth_gb_per_sec=3741,
        hbm_to_vmem_latency_ns=645,
        vmem_to_hbm_latency_ns=512,
        matmul_latency_cycles_by_dtype=immutabledict({
            jnp.float8_e4m3fn: 211,
            jnp.float8_e5m2: 211,
            jnp.bfloat16: 235,
            jnp.float32: 227,
        }),
        default_p_state=3,
        default_vmem_limit_kib=96 * 1024,
    ),
    PlatformInfo(
        chip_version=pltpu.ChipVersion.TPU_8T,
        default_internal_scratch_bytes=278528,
        matmul_cadence_cycles_by_dtype=immutabledict({
            jnp.int4: 16 / 3,
            jnp.float4_e2m1fn: 16 / 3,
            jnp.float8_e4m3fn: 16 / 3,
            jnp.float8_e5m2: 16 / 3,
            jnp.bfloat16: 16,
            jnp.float32: 8,
        }),
        latch_cadence_cycles_by_dtype=immutabledict({
            jnp.int4: 4,
            jnp.float4_e2m1fn: 4,
            jnp.float8_e4m3fn: 4,
            jnp.float8_e5m2: 4,
            jnp.bfloat16: 4,
            jnp.float32: 2,
        }),
        clock_speed_ghz_by_p_state=immutabledict({
            0: 1.7,
            1: 1.8,
            2: 1.9,
            3: 2.0,
            4: 2.05,
        }),
        hbm_to_vmem_bandwidth_gb_per_sec=5679,
        hbm_to_vmem_latency_ns=821,
        vmem_to_hbm_latency_ns=821,
        matmul_latency_cycles_by_dtype=immutabledict({
            jnp.int4: 225,
            jnp.float4_e2m1fn: 225,
            jnp.float8_e4m3fn: 225,
            jnp.float8_e5m2: 225,
            jnp.bfloat16: 243,
            jnp.float32: 235,
        }),
        default_p_state=1,
        default_vmem_limit_kib=32 * 1024,
    ),
)


_PLATFORM_INFO_BY_PLATFORM = immutabledict(
    {platform.chip_version: platform for platform in _PLATFORM_INFOS}
)


def get_platform_info(
    chip_version: pltpu.ChipVersion | None = None,
) -> PlatformInfo:
  """Returns the platform info for the current TPU platform."""
  if chip_version is None:
    chip_version = pltpu.get_tpu_info().chip_version
  return _PLATFORM_INFO_BY_PLATFORM[chip_version]


def get_default_internal_scratch_bytes(
    chip_version: pltpu.ChipVersion | None = None,
) -> int:
  """Returns the minimum internal scratch size in bytes for the platform."""
  return get_platform_info(chip_version).default_internal_scratch_bytes


def get_default_vmem_limit_kib(
    chip_version: pltpu.ChipVersion | None = None,
) -> int:
  """Returns the default VMEM limit in KiB for the platform."""
  return get_platform_info(chip_version).default_vmem_limit_kib
