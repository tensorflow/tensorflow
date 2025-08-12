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
        network_throughput_bytes_per_sec: 20765736
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 13787160
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 8344198
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
        network_throughput_bytes_per_sec: 32064128
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 15873015
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 18593840
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
        network_throughput_bytes_per_sec: 47904191
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 28243601
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 11838697
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 10571522
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 7492390
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 7626310
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
        network_throughput_bytes_per_sec: 56939501
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 90523338
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 122137404
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
        network_throughput_bytes_per_sec: 58447488
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 104746317
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 134736842
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
        network_throughput_bytes_per_sec: 55220017
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 51990251
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 154216867
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 53691275
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 90267983
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 66528066
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
        network_throughput_bytes_per_sec: 125984251
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 127617148
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 221837088
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
        network_throughput_bytes_per_sec: 140814081
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 225749559
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 209492635
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
        network_throughput_bytes_per_sec: 136606189
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 160200250
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 342245989
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 101426307
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 156670746
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 70368334
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
        network_throughput_bytes_per_sec: 253968253
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 228980322
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 254726368
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
        network_throughput_bytes_per_sec: 162539682
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 275565123
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 709141274
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
        network_throughput_bytes_per_sec: 277657266
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 343163538
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 725212464
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 251226692
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 280087527
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 536687631
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
        network_throughput_bytes_per_sec: 471889400
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 519796954
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 676354029
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
        network_throughput_bytes_per_sec: 532224532
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1160997732
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1406593406
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
        network_throughput_bytes_per_sec: 497570456
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 537250786
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1142857142
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 503937007
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 393543428
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 756277695
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
        network_throughput_bytes_per_sec: 870008496
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1400820793
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2285714285
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
        network_throughput_bytes_per_sec: 1777777777
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2343249427
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2813186813
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
        network_throughput_bytes_per_sec: 822489959
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 722143864
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 867796610
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 503937007
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 1109425785
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 713091922
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
        network_throughput_bytes_per_sec: 1650282030
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 3631205673
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 4302521008
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
        network_throughput_bytes_per_sec: 3524956970
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 4729792147
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 5535135135
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
        network_throughput_bytes_per_sec: 1740016992
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 4511013215
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 3605633802
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2763832658
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 2167195767
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 3778597785
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
        network_throughput_bytes_per_sec: 3289959839
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 5031941031
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 7816793893
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
        network_throughput_bytes_per_sec: 4437703141
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 7262411347
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 7876923076
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
        network_throughput_bytes_per_sec: 5859799713
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 3602462620
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 6942372881
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 3599297012
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 4075621890
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 5298835705
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
        network_throughput_bytes_per_sec: 10014669926
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 13107200000
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 16031311154
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
        network_throughput_bytes_per_sec: 7953398058
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 12720496894
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 20739240506
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
        network_throughput_bytes_per_sec: 9173572228
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 17031185031
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 8933478735
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 10317380352
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 6913080168
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 5970845481
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
        network_throughput_bytes_per_sec: 12593389700
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 12328066215
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 27125827814
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
        network_throughput_bytes_per_sec: 27443886097
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 32125490196
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 35158798283
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
        network_throughput_bytes_per_sec: 12681114551
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 18265328874
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 26047694753
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 8650475184
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 13826160337
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 27629005059
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
        network_throughput_bytes_per_sec: 20189772027
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 24129602356
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 50027480916
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
        network_throughput_bytes_per_sec: 33677286742
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 51200000000
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 59795620437
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
        network_throughput_bytes_per_sec: 51280125195
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 36008791208
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 66737270875
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 19106705539
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 30681647940
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 28972590627
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
        network_throughput_bytes_per_sec: 36612290502
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 35463203463
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 47524292965
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
        network_throughput_bytes_per_sec: 66264914054
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 73388577827
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 82022528160
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
        network_throughput_bytes_per_sec: 60963720930
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 93090909090
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 68266666666
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 83591836734
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 58514285714
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 42833986928
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
        network_throughput_bytes_per_sec: 61077353215
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 70888047593
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 97234421364
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
        network_throughput_bytes_per_sec: 88983027834
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 99447647951
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 136818371607
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
        network_throughput_bytes_per_sec: 117658886894
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 98847662141
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 132262361251
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 112123182207
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 79054282267
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 75112893982
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
        network_throughput_bytes_per_sec: 80117359413
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 98698795180
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 115076382791
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
        network_throughput_bytes_per_sec: 128944417117
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 147936794582
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 208547334924
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
        network_throughput_bytes_per_sec: 141470048569
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 153210987726
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 196509745127
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 118402890695
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 106649308380
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 177364005412
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
        network_throughput_bytes_per_sec: 115481938325
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 109821533305
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 165913924050
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
        network_throughput_bytes_per_sec: 204003112840
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 239729309556
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 277989395546
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
        network_throughput_bytes_per_sec: 193750184774
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 240388812471
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 246607714016
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 158875151515
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 191345985401
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 211321241434
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
        network_throughput_bytes_per_sec: 148544553052
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 153772693943
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 178663486113
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
        network_throughput_bytes_per_sec: 259355923818
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 296124258683
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
              num_replica_groups: 4
              num_devices_per_group: 2
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
        network_throughput_bytes_per_sec: 251276300023
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 271581455581
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 266001014713
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 253218063269
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 247189061763
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 317942995755
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
        network_throughput_bytes_per_sec: 176691549414
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 170016376165
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 203192713884
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
        network_throughput_bytes_per_sec: 314227150134
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 342671895424
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 416680309954
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
        network_throughput_bytes_per_sec: 306960187353
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 299037786967
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 303847000869
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 306825457205
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 288545954870
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 395838429596
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
        network_throughput_bytes_per_sec: 200281921497
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 181823478411
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 216201237113
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
        network_throughput_bytes_per_sec: 343936367363
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 383321513434
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 454420801733
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
        network_throughput_bytes_per_sec: 326151166407
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 331854102381
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 330312175145
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 325265917022
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 374926611245
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 435771844155
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
        network_throughput_bytes_per_sec: 212665939916
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 189098712833
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 227204246905
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
        network_throughput_bytes_per_sec: 368082843352
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 402872346556
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 488704223711
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
        network_throughput_bytes_per_sec: 348118354981
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 346765656649
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 347397523501
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 353651264755
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 400219847328
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 519129154031
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
        network_throughput_bytes_per_sec: 220787703321
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 193319306331
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 232468006096
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
        network_throughput_bytes_per_sec: 385603346434
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 423399772870
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 511594072086
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
        network_throughput_bytes_per_sec: 358832552668
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 360350874178
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 365819545593
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 367582840366
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 421983399567
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 558905190219
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
        network_throughput_bytes_per_sec: 227444546849
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 198288807469
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 242242337131
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
        network_throughput_bytes_per_sec: 393268230936
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 434192960662
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 537782991954
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
        network_throughput_bytes_per_sec: 367611030160
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 371222516014
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 389556301155
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 383957524230
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 450461571507
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 623839068919
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
        network_throughput_bytes_per_sec: 229968237690
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 201566254276
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 250289469463
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
        network_throughput_bytes_per_sec: 401897616481
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 443094410881
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 549964466006
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
        network_throughput_bytes_per_sec: 376090719201
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
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 383183624061
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
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 410431685299
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 388143599946
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 468722523974
      }
      entries {
        instruction {
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          dimensions: 0
          channel_id: 1
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        network_throughput_bytes_per_sec: 642663627744
      }
    }
  }
)pb";

#endif  // XLA_SERVICE_GPU_MODEL_COLLECTIVE_INTERPOLATOR_DATA_H_
