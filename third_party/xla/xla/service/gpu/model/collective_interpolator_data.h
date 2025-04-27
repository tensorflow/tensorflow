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

#ifndef XLA_SERVICE_GPU_MODEL_COLLECTIVE_INTERPOLATOR_DATA_H_
#define XLA_SERVICE_GPU_MODEL_COLLECTIVE_INTERPOLATOR_DATA_H_

// Textproto below is generated via
//
//   bazel run --config=cuda -- //xla/tools:collective_perf_table_gen_main
constexpr char kDefaultCollectivePTable[] = R"pb(
  entries {
    key: "sm_90"
    value {
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 28571428
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 158415841
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 33023735
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 167539267
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 32
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 49535603
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 164948453
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 47024246
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 329896907
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 68965517
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 328205128
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 64
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 67297581
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 326530612
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 113174182
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 646464646
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 112775330
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 684491978
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 128
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 240601503
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 656410256
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 345479082
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1333333333
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 266666666
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1333333333
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 451499118
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1280000000
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 413570274
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2680628272
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 708160442
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2534653465
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 666666666
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2652849740
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 739350180
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 5145728643
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1723905723
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 5224489795
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1479768786
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 5305699481
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1472322070
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 10502564102
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 3424749163
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 10395939086
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2314124293
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 10343434343
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 3444911690
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 20480000000
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 6714754098
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 21005128205
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2862334032
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 20897959183
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 5957818181
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 42226804123
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 13255663430
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 42010256410
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 8224899598
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 40554455445
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 18984936268
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 81920000000
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 20177339901
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 81920000000
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 16532795156
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 80313725490
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 22443835616
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 152409302325
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 42666666666
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 151703703703
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 35121114683
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 148945454545
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 43030860144
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 296542986425
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 62594078319
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 295207207207
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 82643127364
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 291271111111
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 63906387128
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 553046413502
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 87323117921
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 537180327868
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 116924174843
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 534987755102
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 74451576256
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 865161716171
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 134432820512
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 826952681388
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 153390286717
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 834853503184
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 110353188802
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 981812734082
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 194613214550
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 934559714795
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 204560280920
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 948079566003
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 144272977435
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 992969696969
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 232397163120
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1005346116970
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 233900513049
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1009216554379
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 165927051190
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1149754385964
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 313428784934
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1141617855198
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 311519904931
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1137901247965
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 195347398817
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1228200292825
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 338032237266
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1230001173020
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 327500897946
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1227481416447
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 208692606229
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1278361475160
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 368002105724
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1276222120797
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 347901791639
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1275833916349
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 218058669855
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1287781393920
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 384032229267
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1287880248714
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 359209009549
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1286991101564
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 222030980976
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1305416744475
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 391036278245
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1303236571251
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 364916444628
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1304452513314
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 224423344971
      }
      entries {
        instruction {
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1315576326674
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 400124397805
      }
      entries {
        instruction {
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1316944621060
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 373842693108
      }
      entries {
        instruction {
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          use_global_device_ids: true
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 8
              num_devices_per_group: 1
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1315318476705
      }
    }
  }
)pb";

#endif  // XLA_SERVICE_GPU_MODEL_COLLECTIVE_INTERPOLATOR_DATA_H_
