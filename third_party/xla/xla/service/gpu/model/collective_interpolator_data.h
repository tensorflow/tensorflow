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
//
// BEGIN_DEFAULT_PERF_TABLE
constexpr char kDefaultCollectivePTable[] = R"pb(
  entries {
    key: "sm_100"
    value {
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "82724b215353fdf447c8f5867b927fe2"
        network_throughput_bytes_per_sec: 14185281385
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "70a080ed258662e7a7c448a580386531"
        network_throughput_bytes_per_sec: 155528554
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "8c4f72b22cf1c427b5192fde2275b82d"
        network_throughput_bytes_per_sec: 75804898766
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b119f67d214e8219e6b672422c7ff82d"
        network_throughput_bytes_per_sec: 4940892641
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "30228c58f0f8bfd498c30a0b4c75491e"
        network_throughput_bytes_per_sec: 523521123354
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "6a6da3f8a701c6de63d3f3eff5a326d0"
        network_throughput_bytes_per_sec: 307692307
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4f39fbed3ed5b26fd8f01ceb12a6958e"
        network_throughput_bytes_per_sec: 11359602
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d99a31a5731cd087d2a57a04dbeda416"
        network_throughput_bytes_per_sec: 277694915254
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "991c99cf49ca7ed01041e4151f354da3"
        network_throughput_bytes_per_sec: 71111111
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "8dd875a32ca9280cac2fb6c8b4a3f900"
        network_throughput_bytes_per_sec: 2458583433
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "36c99c9ecd2afb910616dba7e7604d76"
        network_throughput_bytes_per_sec: 1120350109
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "9efe18a44677d5ebf6d950d12f0105d0"
        network_throughput_bytes_per_sec: 129005394058
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b6cdb267bfc64ec38e1e2740095c8805"
        network_throughput_bytes_per_sec: 7111111111
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "27a51e9c9148298fd01ee900e6a81c2c"
        network_throughput_bytes_per_sec: 851808285946
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "22444f0e312a3499cbfd75eaf67c0888"
        network_throughput_bytes_per_sec: 127937530502
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "bee86716cff01212fb687be758fc96e1"
        network_throughput_bytes_per_sec: 520870550009
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e67540b745e062c37cb2d5e38a645a43"
        network_throughput_bytes_per_sec: 363836224843
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "02f84dcfcb5697b10aa0548ff15c1379"
        network_throughput_bytes_per_sec: 7074265975
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "260f965f1a4678225622c8bb1bb605bd"
        network_throughput_bytes_per_sec: 117960007874
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "f6f986332989bc7ea465794f1c2b6856"
        network_throughput_bytes_per_sec: 536489870069
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4fd09bd390682728f7da420003fb7c37"
        network_throughput_bytes_per_sec: 533889468
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f9d967d15c65b7c80d2055c0c6dbf3c6"
        network_throughput_bytes_per_sec: 333722196805
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "bb6b916563714f9bbad245e71a3d56e3"
        network_throughput_bytes_per_sec: 32031280547
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c04d9b7c7ac56f98fcf87ee9c131ab68"
        network_throughput_bytes_per_sec: 733454255330
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ef5d4bc6c48c17023f1713afce3c88c3"
        network_throughput_bytes_per_sec: 26947368421
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "644ed15c889d04ca582b384ff68fc71e"
        network_throughput_bytes_per_sec: 380952380
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "5b2d32958f53fbfe8142848551afad7c"
        network_throughput_bytes_per_sec: 9683215130
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "498af7a3213702edfadb727477672515"
        network_throughput_bytes_per_sec: 25051987767
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "6b3fd8cf011b133409ba2ce19f78aed7"
        network_throughput_bytes_per_sec: 6239146991
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f9040b578d0f9eba41c4ac1a07ee4224"
        network_throughput_bytes_per_sec: 44582312925
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "ea38ae3f2a296149dde0cac2c673fd8f"
        network_throughput_bytes_per_sec: 429631462026
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "8154a3ab4411af0b0a94fc14f69ac096"
        network_throughput_bytes_per_sec: 101057825751
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "64fc8a1589366b418b88651876990852"
        network_throughput_bytes_per_sec: 590414414414
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "8abe97935f10f37b31406be7fe615de0"
        network_throughput_bytes_per_sec: 262026612
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "61c2170cfeaf7234b58eea7d3c5cc136"
        network_throughput_bytes_per_sec: 12709395908
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "9f6af0d6b827293d0386ca2378e5d71b"
        network_throughput_bytes_per_sec: 22743425
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b8e3907c6dfb227acf1602dcedd1dfae"
        network_throughput_bytes_per_sec: 695270135305
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "83d2f5e591feefe0553a301d532b898f"
        network_throughput_bytes_per_sec: 337325398101
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "413eb782ec4e2c4409f95026e7b720e6"
        network_throughput_bytes_per_sec: 48617210682
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ca28c1cb14c2361c8daf298f6de5fc47"
        network_throughput_bytes_per_sec: 102480062548
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c8859ac88de21b0d40acf7b94c89a34e"
        network_throughput_bytes_per_sec: 757137293394
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e5dc26fcdbdb577aa5941155f0d4fc57"
        network_throughput_bytes_per_sec: 10252816020
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "85bb8349a62442dcab56384b99cbe6d0"
        network_throughput_bytes_per_sec: 511875030510
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "6632fafd05450c9ca5bdd86c67f7cc0a"
        network_throughput_bytes_per_sec: 1452482269
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c6fbc0a09d2d44806949eef49196c7a2"
        network_throughput_bytes_per_sec: 639200998
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "4ffa457833b6c4ff6e5147781e31302a"
        network_throughput_bytes_per_sec: 113817915388
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ad9a13d7c03557ae4a78b547c910aaa9"
        network_throughput_bytes_per_sec: 18244988864
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "2e4edd9d5f901a539189122797458ea4"
        network_throughput_bytes_per_sec: 24526946107
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e39d4c38abe32c0b8bf790196f492d26"
        network_throughput_bytes_per_sec: 383216445865
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ca3eb58708ad6e9c3551f90b9d193653"
        network_throughput_bytes_per_sec: 227654363873
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0f527662a7b68694b48d50d10a297e0b"
        network_throughput_bytes_per_sec: 3205007824
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4351be00ad096ee1fcfe565c2215c7dd"
        network_throughput_bytes_per_sec: 693502645502
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "6ab6cfdfc119a9143b046bd2262766d6"
        network_throughput_bytes_per_sec: 102687525
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "aa53481ae940f8be73d0809b81bd85ee"
        network_throughput_bytes_per_sec: 28971790125
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "84086fea224a69018a6bcf0db282b861"
        network_throughput_bytes_per_sec: 584490523968
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "695c9c2e1a16cd287cd6b80d66c3cf24"
        network_throughput_bytes_per_sec: 170638893409
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0120a2fd4590718b617dbde5030314f0"
        network_throughput_bytes_per_sec: 111888111
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b398f3b5618fef0e8beefc3d9fb45eee"
        network_throughput_bytes_per_sec: 292082451253
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "a9f1ccc1dcdd0ea4b6d345c626fc0464"
        network_throughput_bytes_per_sec: 27777777
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "44d6bfd785c449cc8640234e220f77aa"
        network_throughput_bytes_per_sec: 10613547107
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b51292fa84bc7fc3b5a42c808ed0538a"
        network_throughput_bytes_per_sec: 650456170278
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "19bfe133a0d02c4cac3fe71ed6e3e741"
        network_throughput_bytes_per_sec: 273208963001
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d04da75eb576bed3d4db82103261bf34"
        network_throughput_bytes_per_sec: 6370139968
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c986af5a1df1d20f73d7d40cf5b1e067"
        network_throughput_bytes_per_sec: 1066666666
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d71debdb12a8986a0f288def2a8ac093"
        network_throughput_bytes_per_sec: 390095238095
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d6ceb25936203837f994d4ea62fccbcb"
        network_throughput_bytes_per_sec: 228367528
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "a83d894078ce2cfadd898cfbdd4955ea"
        network_throughput_bytes_per_sec: 16094302554
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "a9e70188f014a7fd5d3664cb93a8ceea"
        network_throughput_bytes_per_sec: 333550488
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "04a5265203d9bcb5e98c819340f31d6c"
        network_throughput_bytes_per_sec: 66651369003
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "99043b67c066b466fdfabf4ba0e10d9d"
        network_throughput_bytes_per_sec: 246925488
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "10f1eac7685082516c77e28c1c570603"
        network_throughput_bytes_per_sec: 749183531303
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "62c4fc6430ca1eb1c9da917231cf7c2c"
        network_throughput_bytes_per_sec: 424438777575
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f7fd5b7bdf4d97b0eb10f5fbab3117c5"
        network_throughput_bytes_per_sec: 14234578627
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "a5c25d90d3703c3e05a5428a5fbafe10"
        network_throughput_bytes_per_sec: 1961450975
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "35882d22990344fadfe4e45b8e2721eb"
        network_throughput_bytes_per_sec: 478822324015
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "0f9cdfcb5a2647c85e89d23875e02e61"
        network_throughput_bytes_per_sec: 70450636
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "71fef15b131813a4d472ccf5528d373b"
        network_throughput_bytes_per_sec: 210526315
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c9746f1c866d390a80ecaa1cd0747467"
        network_throughput_bytes_per_sec: 213494044589
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d0a5ced62829c6ceb591eb442eb1b79c"
        network_throughput_bytes_per_sec: 653061224
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7090d041bc0599ce9b8dc0095e8d7135"
        network_throughput_bytes_per_sec: 34168925964
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "584651046fab6aa022a9fcdaa741ca49"
        network_throughput_bytes_per_sec: 4556173526
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "e78c4bfa48ee5d7f81743a050aa5f803"
        network_throughput_bytes_per_sec: 79559438818
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "3e00e3751db2f0a54edfc160509a0c32"
        network_throughput_bytes_per_sec: 7185964912
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "df2f19ebe8fc637197e39b52d97a794c"
        network_throughput_bytes_per_sec: 5143911149
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "3fbaf73ace028a5c0673748316b980bf"
        network_throughput_bytes_per_sec: 331029083303
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "28ed8d8831773650ac1210294feb985d"
        network_throughput_bytes_per_sec: 1213270142
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "6411603f6b84e86918c660203d2586d6"
        network_throughput_bytes_per_sec: 138988802
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "361bab01e1544bbd7d4c57964c7cb2e8"
        network_throughput_bytes_per_sec: 296124258683
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f2e7c307868389c7b945a60985fdbfb7"
        network_throughput_bytes_per_sec: 600473013600
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d8ba4d1f1855930537e677def956da0c"
        network_throughput_bytes_per_sec: 85333333333
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "21c3dea5239284aed45517eccc2c77a1"
        network_throughput_bytes_per_sec: 12118916
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b5ef73f8707a38b85661790207aa156e"
        network_throughput_bytes_per_sec: 704488436788
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e188b963e5912029be041b83f8c32803"
        network_throughput_bytes_per_sec: 130031746031
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "58212b0c758c8c906c2df8d9cd23841e"
        network_throughput_bytes_per_sec: 675411272141
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "aedd2df037a65be1250f90a94291ed61"
        network_throughput_bytes_per_sec: 11394345076
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "b01e85aab5ef3da6357ca6f9cfb67b8f"
        network_throughput_bytes_per_sec: 99244105294
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "3fc3265f1d4f48b553f24df733a0ec07"
        network_throughput_bytes_per_sec: 137931034
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "905de60dd6e7b89f580e55ae80ac8d79"
        network_throughput_bytes_per_sec: 330989898989
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "923306d188529fd23828978fba917eca"
        network_throughput_bytes_per_sec: 1007564110
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "012c7d96d1729d9e95f6cb6f9cc6646d"
        network_throughput_bytes_per_sec: 1183815028
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "add1d98ea03d7ecc59ff9877b5bd5e93"
        network_throughput_bytes_per_sec: 203567462628
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "8f93e3c7983b80f6171d861fa67a2bb5"
        network_throughput_bytes_per_sec: 2135557872
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "8370a1824c7b75672651c80e67bfcc33"
        network_throughput_bytes_per_sec: 671948734380
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "faccbf0108d642668cb20ee319a39541"
        network_throughput_bytes_per_sec: 510805041535
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "fc2857874212cf751e4b60decec734a7"
        network_throughput_bytes_per_sec: 52588331
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "eebdac40c94b93e0e8351eba013b7958"
        network_throughput_bytes_per_sec: 44826265389
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "5357a763f8b75abbd8c2b3aa99d399b0"
        network_throughput_bytes_per_sec: 147976878
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "00842feada0344771c1b4e414c197917"
        network_throughput_bytes_per_sec: 513185355119
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "25f7485f5479f664a4f493d61ff1a4d8"
        network_throughput_bytes_per_sec: 5207485
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ea64b110db7c46aafbb394dd547e9e23"
        network_throughput_bytes_per_sec: 804836343575
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "766a46b63049615920fa933700606ad3"
        network_throughput_bytes_per_sec: 4003910068
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "a9dda3180d9de81b8ac47f5af4e3717c"
        network_throughput_bytes_per_sec: 127023506
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4f6b04c57baf41d308928831e49e6f05"
        network_throughput_bytes_per_sec: 6924767540
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "a80bdc2e68784e2f6b048f9070199b93"
        network_throughput_bytes_per_sec: 1185185185
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "e824c0c6cda89c864c487464c4714920"
        network_throughput_bytes_per_sec: 132441685004
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 128
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "22006dffcceb6352ce0d7f47b568a045"
        network_throughput_bytes_per_sec: 5436629
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ae02e28f43d17a07b5679b487f20fb4a"
        network_throughput_bytes_per_sec: 157349339735
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ef648be80ef0366d556323938cba8b8b"
        network_throughput_bytes_per_sec: 145797552836
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4a53acb196d9813effc6e5c5955a28c2"
        network_throughput_bytes_per_sec: 3744058500
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "f08943c7fb325a80c9aedfc3d9219cfb"
        network_throughput_bytes_per_sec: 935159817
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "cc87abd4e4f49d00a84464d67ac1cc63"
        network_throughput_bytes_per_sec: 26947368421
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b064bd92e8bb26128984b39785b63827"
        network_throughput_bytes_per_sec: 21375081539
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "5cfc87185d90d1302586da3e1c0f6fde"
        network_throughput_bytes_per_sec: 1067778936
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e6fb7f1db9f0def5ca27f01f28291a63"
        network_throughput_bytes_per_sec: 842105263
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "9c002f2bdf400f2638aac27df778dfc1"
        network_throughput_bytes_per_sec: 438013106023
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b9ca069b3d1f1eeefa7a6ecf54baacae"
        network_throughput_bytes_per_sec: 848534088610
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "9ca39fa8794daa9631163ee0e266097e"
        network_throughput_bytes_per_sec: 453900709
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7137b6e88084dce1309b8c65093ae1ff"
        network_throughput_bytes_per_sec: 938598637743
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d08ba83901c2428a2cacfc0d6e826840"
        network_throughput_bytes_per_sec: 293924597056
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c2aa9452829d26d4fbfc6be7dcd22902"
        network_throughput_bytes_per_sec: 638305280779
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4ce7115cb0355a0436faffe9d4f63a60"
        network_throughput_bytes_per_sec: 49960967
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b253ab825e72a17754e0f920ba0ac47b"
        network_throughput_bytes_per_sec: 619725768321
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "90fc6688a7894e5c0b2688b8b3b56e5b"
        network_throughput_bytes_per_sec: 1204705882
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0c83ea7171fd3e561e96f900b77b901b"
        network_throughput_bytes_per_sec: 34026998961
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "5ecb8bfdacc065c41d076a95fde10de2"
        network_throughput_bytes_per_sec: 51360501567
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "98b831c4ca72406f37389214c5e19865"
        network_throughput_bytes_per_sec: 311496769402
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "224b0c65bfb5a0772e3b3ef5e529c630"
        network_throughput_bytes_per_sec: 20505632040
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7c15270553f884594396ea3a9e22288a"
        network_throughput_bytes_per_sec: 208464413518
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b5dc833e1006db332dcf16fc073558e4"
        network_throughput_bytes_per_sec: 215578947368
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "0cb65c209a26b3c7c2d519854b62c67c"
        network_throughput_bytes_per_sec: 8162157113
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7cac54d6e1d08a6cd127ab0f81c94048"
        network_throughput_bytes_per_sec: 608355065632
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f40071ad502637ac5fbb867e29baea24"
        network_throughput_bytes_per_sec: 37037037
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "a7adb6a1534bbf74d4512d8e05eb5ad0"
        network_throughput_bytes_per_sec: 859356451365
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "96b26a3b1e59d77bab95e0026d23275c"
        network_throughput_bytes_per_sec: 2074974670
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7607dfea803f89904d8e152fb8189956"
        network_throughput_bytes_per_sec: 4271115745
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "25a76511510793444ae58b29d1d310cd"
        network_throughput_bytes_per_sec: 612396554241
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "17e8b701cba039ffed5c4c64b81bd38e"
        network_throughput_bytes_per_sec: 50567901234
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "f6d54f4055232256fcd6c0bf18d19f73"
        network_throughput_bytes_per_sec: 155498651441
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "1f93b4ce5f502f4229278ea4c2936bb8"
        network_throughput_bytes_per_sec: 8393442622
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "01cbebe328fc80e3c97d6b476fec64ff"
        network_throughput_bytes_per_sec: 228571428
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "abd8c8f7ce2fd95b63645316537b74e7"
        network_throughput_bytes_per_sec: 528429514717
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "bac7b1d59bfbf3ce3c9770a882b7d9d9"
        network_throughput_bytes_per_sec: 1057851239
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4afbc2a251fde8fe2a2f21f9b3680acf"
        network_throughput_bytes_per_sec: 12700775193
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0f6fd7c2255ef5d6619d3e266e4493b1"
        network_throughput_bytes_per_sec: 162017305315
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b3774b957d17024bdec4dfb6aa7015b8"
        network_throughput_bytes_per_sec: 10680573663
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "d1ed25619ddb7091e280b64f8bf17b71"
        network_throughput_bytes_per_sec: 499055741
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d23936a34187f9cecae83b71e3b7c071"
        network_throughput_bytes_per_sec: 363894811
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0ac37c0c61461a98a53778858abef7c2"
        network_throughput_bytes_per_sec: 1773160173
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c5c9c12452980db70f61be1deb2f609c"
        network_throughput_bytes_per_sec: 4000000000
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "9ea76d324a2a87d6007857f1bbd58e8b"
        network_throughput_bytes_per_sec: 6557534520
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e001496ab38237a7a240b9087b52ce67"
        network_throughput_bytes_per_sec: 759837681159
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 128
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "687f7f1c2f260eab74a9b0fe20926d8a"
        network_throughput_bytes_per_sec: 100000000
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "7d42a2c52405b0487725b632c32b3246"
        network_throughput_bytes_per_sec: 56303161285
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4fcbe7e7d7bb2d20520c051dd60ad89e"
        network_throughput_bytes_per_sec: 118832275611
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "713bd2c46abfb6ce361f71da4eb7d023"
        network_throughput_bytes_per_sec: 75851851851
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "a2d264c6a62d0909b4dc78be82a6aff6"
        network_throughput_bytes_per_sec: 124878048
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "03e84199997302127c56cdb92f61f92d"
        network_throughput_bytes_per_sec: 218271440466
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "64fcb07cb34dfdc261edff3094f5e329"
        network_throughput_bytes_per_sec: 675180232207
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "9f83571c52a45ed6fa719a3ef1b35f56"
        network_throughput_bytes_per_sec: 189910979
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "e70429d2d9bdfaadf7c6d6cd451ad2d0"
        network_throughput_bytes_per_sec: 150433377
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "378df9b975bd6f83307c54c5ebf8a5e3"
        network_throughput_bytes_per_sec: 7192273924
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "50e0778e29cbb85bbe9e52c8a9e3f53b"
        network_throughput_bytes_per_sec: 89134308058
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "fd47bb6bf2e69ef941ae3455e0980ec2"
        network_throughput_bytes_per_sec: 68195629552
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "70cc3d28822cfa34a6f4f77936bd4122"
        network_throughput_bytes_per_sec: 567411255411
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7e46aa6a95b79c94bd27c5e9de8038c1"
        network_throughput_bytes_per_sec: 726223530430
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "906601acb6c0bdb8e772d3adb5a7e148"
        network_throughput_bytes_per_sec: 318474107820
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "d1dbd5305c09a5bbfd5a1e9c7c51dd3f"
        network_throughput_bytes_per_sec: 269367354
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "29421a4f1a4df068f67122b659311863"
        network_throughput_bytes_per_sec: 2371742906
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f6813a624e0d4cd5931fd1911a59ce8d"
        network_throughput_bytes_per_sec: 781679389
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d84ec32ec9d5bce065cad02d30309053"
        network_throughput_bytes_per_sec: 413557878130
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "a490a74ef07c6ec14e8318d4b8142f8e"
        network_throughput_bytes_per_sec: 367401723439
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "a6f8b72cb4b9af96eef324934f9de021"
        network_throughput_bytes_per_sec: 22199098
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "561c9e59f33d316b74205823cb42d04d"
        network_throughput_bytes_per_sec: 513444984578
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "668e8b905ffbce1829608cbf4befc332"
        network_throughput_bytes_per_sec: 85333333333
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "48e28feb64dc3689b568926698d8e02e"
        network_throughput_bytes_per_sec: 5913879
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0690a7935a759eb144aeec77b49771d9"
        network_throughput_bytes_per_sec: 51160031225
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "5d605c04268d15a46391c0ef5400e98e"
        network_throughput_bytes_per_sec: 128000000000
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "3aa698dd32575d30602a31ddba9fefac"
        network_throughput_bytes_per_sec: 38120333006
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "3f33c91a674d3a639ad8460a9fee21f0"
        network_throughput_bytes_per_sec: 76830732
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ea826b5b3a3cd11ff917bb395a93cd6f"
        network_throughput_bytes_per_sec: 524288000000
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d5bb5db4849bbe872fc0a3ab1ec6d0d9"
        network_throughput_bytes_per_sec: 3657142857
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4447fc843a198e997259f2a45d1c5078"
        network_throughput_bytes_per_sec: 515460734914
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7e50a143617aca892c3824cf04c47087"
        network_throughput_bytes_per_sec: 461495736370
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 64
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "790112769afed6a30fcc1c8cbce08768"
        network_throughput_bytes_per_sec: 38415366
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e77afb055f2c55ce8a9f881f93f4ccec"
        network_throughput_bytes_per_sec: 592290333968
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "6dd3b94c1709ebff95103a4422009c8b"
        network_throughput_bytes_per_sec: 413455961358
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c708aebbaea40d6ad370faaa7d411d0f"
        network_throughput_bytes_per_sec: 115481938325
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "9c2db8231f5ef24a566554bcd16a60d4"
        network_throughput_bytes_per_sec: 626225822104
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "b9f629195d1e0d1afeebe40aef9955df"
        network_throughput_bytes_per_sec: 73253995144
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "bffcfdab46001339d858d2bde0841588"
        network_throughput_bytes_per_sec: 12424771
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d4bb9864d0881eafd2fa5efec09c6e99"
        network_throughput_bytes_per_sec: 50039093
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "d9af6aa7b045746dbcb1015482010070"
        network_throughput_bytes_per_sec: 26750752589
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "7c2845488c50871c6df2b01b09b93607"
        network_throughput_bytes_per_sec: 524681639325
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "275f04921ccc4b0fa29127fb3a80a2f2"
        network_throughput_bytes_per_sec: 2967175261
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "abd659445180bd7e7b25417a2c970841"
        network_throughput_bytes_per_sec: 220474348191
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "696f90418e9a91559c454b6ef6ce9b85"
        network_throughput_bytes_per_sec: 6948261238
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "3007a47f1e00d56e53224073fd790288"
        network_throughput_bytes_per_sec: 1878899082
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "8eb0c3afb6d4a5a42937daf0b2ca9327"
        network_throughput_bytes_per_sec: 1853393665
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "6823bce11114d8ceb5386584c6072fc8"
        network_throughput_bytes_per_sec: 62972756
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e1c68238c33369569964e0715085e11e"
        network_throughput_bytes_per_sec: 333728835136
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "3c0dab67f668aa96622d925405aa4c35"
        network_throughput_bytes_per_sec: 278053085
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "353ed7150bd91617fa8843f2620c704d"
        network_throughput_bytes_per_sec: 70156207
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "8f9d476f5290f9fd2653c4f975ec810e"
        network_throughput_bytes_per_sec: 28370562770
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "46eb6d6f2c352a68dd5943ccbf21f917"
        network_throughput_bytes_per_sec: 455111111111
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f85a8aefce6c3bcea1643e35008f1774"
        network_throughput_bytes_per_sec: 390822213939
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b0b00085d45064632771977a738377fa"
        network_throughput_bytes_per_sec: 405874201664
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "75c0063e60c072af0ddaa5e2dbbfa741"
        network_throughput_bytes_per_sec: 356050069
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ade554f65ab408a90ce8976d623c021f"
        network_throughput_bytes_per_sec: 25680250783
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "d28e01182634547a1027d109446f3b88"
        network_throughput_bytes_per_sec: 4882729846
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "a69547b3c276d3341642608b1db494a6"
        network_throughput_bytes_per_sec: 91511754502
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "baf06cabffc7267e4ba88f3f0469f867"
        network_throughput_bytes_per_sec: 538168440
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c37087da0a44968200e66995156557e7"
        network_throughput_bytes_per_sec: 399305407463
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "02feb5e94baeeab2372bc39cf64b4dc3"
        network_throughput_bytes_per_sec: 121212121
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "067cb94ec99578ac4a5d635f77d6836e"
        network_throughput_bytes_per_sec: 857217600
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "61804978175d4c86d2e9535c315a7c67"
        network_throughput_bytes_per_sec: 281572502685
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "3b80581304fa3c1ea59b5614ce83e167"
        network_throughput_bytes_per_sec: 4566332218
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "71a05d2710cbb210f44bd79ab02e4544"
        network_throughput_bytes_per_sec: 155802442594
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "87766ec8522f636b65d29ffbbac0f005"
        network_throughput_bytes_per_sec: 2681286310
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "77c8cbc50f987483fee44f7a20bf8b1f"
        network_throughput_bytes_per_sec: 528649357196
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "3b97564ed96cdb745f18dfba342d89a0"
        network_throughput_bytes_per_sec: 198443603330
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "fb071244e8d81a3688446f7a2515f445"
        network_throughput_bytes_per_sec: 633006942348
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "3a320de13af8d0a2b9013cfbf39985fb"
        network_throughput_bytes_per_sec: 295953757
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "60c70a6d99cce4e304e50c40c8f99fce"
        network_throughput_bytes_per_sec: 378611362482
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c8fe77ad4d6a5f9c9e08e64b66c5aa36"
        network_throughput_bytes_per_sec: 2064516129
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "fab7de7fff87c4825d063b15cf576f4a"
        network_throughput_bytes_per_sec: 81632653
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "762f33c509d278627132d4806885cfc9"
        network_throughput_bytes_per_sec: 29283288650
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ad7d779a1ec5f906606de9e4e93650d9"
        network_throughput_bytes_per_sec: 1601250977
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "dd72a6e820b6d534b73545dc695b9277"
        network_throughput_bytes_per_sec: 1924812030
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "141f0055829b89a154053476d8d5fabe"
        network_throughput_bytes_per_sec: 43690666666
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "96ca5c7e1f1638828edc08d3daa8f8f7"
        network_throughput_bytes_per_sec: 78486227544
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "1c34dad9da64a4a58f0c8f86c14a73a2"
        network_throughput_bytes_per_sec: 30654738934
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 128
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e94911366d0b4cfff0d6742f123be8a9"
        network_throughput_bytes_per_sec: 77575757
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7da45b992c6f90f878e16349e67435ed"
        network_throughput_bytes_per_sec: 818281032044
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "656be27a022b3b74531f5f2327584a2a"
        network_throughput_bytes_per_sec: 3757798165
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "24af8009842c9b30f9fc309675fd46c8"
        network_throughput_bytes_per_sec: 687140235910
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0588b7c974105348d0a2515c0abd898b"
        network_throughput_bytes_per_sec: 99712438189
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "188b7c8f49ee518ef3f12f7239d0542a"
        network_throughput_bytes_per_sec: 8245596376
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "83591db4b7a2726af43dd49fc404007b"
        network_throughput_bytes_per_sec: 132297300387
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "21ea7fae57499ff2b1700818db1ccc37"
        network_throughput_bytes_per_sec: 130470612022
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "afc15c1b4deac0e2789adf42fce916f4"
        network_throughput_bytes_per_sec: 694823821751
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "2965a87c08fc2bb44b5e63f4a8232930"
        network_throughput_bytes_per_sec: 178086956521
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "66a5b4b136e3e63482a26491f2086663"
        network_throughput_bytes_per_sec: 380386329
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ffb987920d3dcd0f23601030220f2c32"
        network_throughput_bytes_per_sec: 515524090462
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "96e82cb9d09f5d9b43c2800f01b5f3ff"
        network_throughput_bytes_per_sec: 280105783357
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "542e81cf1beafb7de263b511ad1f5d7c"
        network_throughput_bytes_per_sec: 77926278240
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "2c9f9cea76ec0a7f35987d8bdeea1d30"
        network_throughput_bytes_per_sec: 3552471812
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7ab9f4f3c725f05b8e8f79d1d9a79a65"
        network_throughput_bytes_per_sec: 890434782
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "03330f0defed011fd622da3ddcd58de5"
        network_throughput_bytes_per_sec: 17636167922
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d51ae4f483b29ba1518794ed1f635d41"
        network_throughput_bytes_per_sec: 351171449502
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "fac36931d73a8f77c6af1f29aa01f950"
        network_throughput_bytes_per_sec: 196952667167
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "954fa4e199cb0e689954013992370dc4"
        network_throughput_bytes_per_sec: 731428571
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "32921213d78db478164e6ece7132d57a"
        network_throughput_bytes_per_sec: 399457523809
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "3a965e23c93556616a155131d28e076f"
        network_throughput_bytes_per_sec: 693387998016
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e937e6f6cd2b6db16fe9cd7c2979d357"
        network_throughput_bytes_per_sec: 66064516129
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "1f4b3db733dab96bb8fa97d8f2bb2c7e"
        network_throughput_bytes_per_sec: 531221766925
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c578baa139f847a50200e65a501cfe37"
        network_throughput_bytes_per_sec: 1456614509
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "997ccdcc3c56a275439d3efb9d75628b"
        network_throughput_bytes_per_sec: 6134969
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ecf9d950bcd7b6c6223401f066216fd4"
        network_throughput_bytes_per_sec: 310303030
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "bdda763fe4cbd4dbd2cd6e538df4d2f5"
        network_throughput_bytes_per_sec: 350752968723
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "59b183e8704a0ba13869ade15fa2b92b"
        network_throughput_bytes_per_sec: 156878515858
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0b76c03ffc5616ae8aaf7fe05d58a8e5"
        network_throughput_bytes_per_sec: 585960324112
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ee0857faea7d71857c6fb8036e979d2d"
        network_throughput_bytes_per_sec: 48725650557
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "393073cf2c2fb7f64008edb89f2f877a"
        network_throughput_bytes_per_sec: 47310939156
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "22d939e4ee859968ac17ebf1c62fef05"
        network_throughput_bytes_per_sec: 715049908366
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "bd85fa7c974409dc2c659688fdfb262f"
        network_throughput_bytes_per_sec: 4413793103
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 64
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f4c16425b37cb7c0469f632bc5f2954f"
        network_throughput_bytes_per_sec: 62256809
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b4c5c1f7997e35be9d661c5ab8917bbc"
        network_throughput_bytes_per_sec: 332036316
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "4a5e09d90eb94f08a4eb31421f1e8443"
        network_throughput_bytes_per_sec: 80874710166
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "96c966eff2db96826fa05a63d70abdfb"
        network_throughput_bytes_per_sec: 293133731697
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "fd886f592d094bb3c992eb050b3aeb7a"
        network_throughput_bytes_per_sec: 76920187793
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "a0c14787b94ec324f39d4b4cde6aaee8"
        network_throughput_bytes_per_sec: 1561049973
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b3181d33b743bf7d8e93edc0102fa54b"
        network_throughput_bytes_per_sec: 577250756950
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c092460e68acec60d687e3ff6c5a6674"
        network_throughput_bytes_per_sec: 788440058273
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ef3bbaf6d14ca5829ae80710febc85eb"
        network_throughput_bytes_per_sec: 500000000
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "15185f0d28c7a7edc1b4aecfa6d8f221"
        network_throughput_bytes_per_sec: 293947770934
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "6b3993ffc7a2464a3f8f42c61f55394d"
        network_throughput_bytes_per_sec: 633198067632
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "1287402cc6f1747882876a7ef488e090"
        network_throughput_bytes_per_sec: 33590978985
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "9baae3a878bfbfff40ac180eb9c74753"
        network_throughput_bytes_per_sec: 16384000000
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "47bea0ecc1ea80cb91e9b291e81894f8"
        network_throughput_bytes_per_sec: 68246672524
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "a9ce40b744201dc85700fe442e18c6d3"
        network_throughput_bytes_per_sec: 46512420156
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ae9b470b1a59d43e5b5fa203ca090cd9"
        network_throughput_bytes_per_sec: 327680000000
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f36d3cb56a18e44b99c8e2bd01e47a20"
        network_throughput_bytes_per_sec: 100000000
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "56f48d98625522ae94180017c6e4235a"
        network_throughput_bytes_per_sec: 592592592
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "53d4043108c4d45128534295dc2c4234"
        network_throughput_bytes_per_sec: 468007312
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "f57f58c135e77c22b55a8bdbe156e2a4"
        network_throughput_bytes_per_sec: 15208656049
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "5016df44b768d5084d1607b448e77a3d"
        network_throughput_bytes_per_sec: 13462612982
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "be567d531287055cea40ba66db60de94"
        network_throughput_bytes_per_sec: 413313362238
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "680f6ed8e838fee643167dc7a214bcd4"
        network_throughput_bytes_per_sec: 344699539776
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "d8c02d04f293873b5f6c56e662530193"
        network_throughput_bytes_per_sec: 883755583649
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "8426634cb954bc9a04f5d3df48489b36"
        network_throughput_bytes_per_sec: 584653470867
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "c5eb925239f355be19b2a2ae51019597"
        network_throughput_bytes_per_sec: 241998128458
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "2d89d55373f92fb4e1b55e7456f04f79"
        network_throughput_bytes_per_sec: 134226318484
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "147048ce1f70eaaf4c41e3a478797c71"
        network_throughput_bytes_per_sec: 648922686
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "c161c6ad6758b19e819c0db57a16b335"
        network_throughput_bytes_per_sec: 6090706319
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0ee3ab31a2cbac43bf62cb9214b7c1d1"
        network_throughput_bytes_per_sec: 102480062548
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "76f465d808cb817683770b3e6ab6838d"
        network_throughput_bytes_per_sec: 92827195467
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "4a4598a1e81e510779605c8c674a3ac0"
        network_throughput_bytes_per_sec: 810630589713
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "1576d013b336b385ef44a95d0cd74a2c"
        network_throughput_bytes_per_sec: 15968810916
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "95b6a84a10fab4f19072622cadfd1acf"
        network_throughput_bytes_per_sec: 2560000000
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "44ab650a56833b85a693cb515c43dee0"
        network_throughput_bytes_per_sec: 2058291457
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "5326b51b78d6cb159c04918bfead91ed"
        network_throughput_bytes_per_sec: 584898061637
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "99623f61db0dd4933df0e1e1215c75eb"
        network_throughput_bytes_per_sec: 124121212121
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "1dc1e161c38f7eff0cea4c9c323dcfad"
        network_throughput_bytes_per_sec: 642214668504
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "6fdb3d7a311222b991ec1edef14c5c26"
        network_throughput_bytes_per_sec: 809086419753
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "f791188a95c2a553c9030fa46c38f2a0"
        network_throughput_bytes_per_sec: 97523809523
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ecaeb13272e9384319989d342680549c"
        network_throughput_bytes_per_sec: 531671858
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "6b9183f8b747bcf1e11a605c2867f7ff"
        network_throughput_bytes_per_sec: 128313264806
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "7c115f6e4ef6c361899235020514a18f"
        network_throughput_bytes_per_sec: 68912723449
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "0516ad83d2a5538970091f77531c73c6"
        network_throughput_bytes_per_sec: 766958445714
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "31a79cde8d9fdcd983bb9252c3c92f73"
        network_throughput_bytes_per_sec: 42622441721
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-reduce"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "48cd9372c015dda07a982da1a5620727"
        network_throughput_bytes_per_sec: 121442125
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "e4c41dd6f65b2df6f78f48e9ef705d3e"
        network_throughput_bytes_per_sec: 468532618409
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "0a1bea9a023ace07a93df0c757a64d8d"
        network_throughput_bytes_per_sec: 3642507781
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "b4310b6629879dd69750c1adf77eca35"
        network_throughput_bytes_per_sec: 20792722
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "22929fda0f320c7393deb2c18af09d78"
        network_throughput_bytes_per_sec: 114695340
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "1f1e13b498f3efcf560154972498ced3"
        network_throughput_bytes_per_sec: 5616005
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 4
              num_devices_per_group: 2
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "5c21cfc6706586931dd655d0d05f90bb"
        network_throughput_bytes_per_sec: 126984126
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-to-all"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 2
              num_devices_per_group: 4
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "15f365ac896c0b583d549fa6577c6d56"
        network_throughput_bytes_per_sec: 266112266
      }
      entries {
        instruction {
          name: "_"
          opcode: "reduce-scatter"
          shape {
            element_type: F32
            dimensions: 32
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 4
          operand_ids: 3
          called_computation_ids: 2
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "ad8d88294e7338062a0ebdb4f4cd2eb5"
        network_throughput_bytes_per_sec: 22727272
      }
      entries {
        instruction {
          name: "_"
          opcode: "all-gather"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          dimensions: 0
          channel_id: 1
          id: 1
          operand_ids: 0
          frontend_attributes {}
          use_global_device_ids: true
          statistics_viz {}
          collective_device_list {
            iota_replica_group_list {
              num_replica_groups: 1
              num_devices_per_group: 8
              iota_reshape_dims: 8
              iota_transpose_perm: 0
            }
          }
        }
        fingerprint: "2168cceecb93680ecc7859d77fc1492b"
        network_throughput_bytes_per_sec: 30303030
      }
    }
  }
  entries {
    key: "sm_90"
    value {
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "df2f19ebe8fc637197e39b52d97a794c"
        network_throughput_bytes_per_sec: 7423231579
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "f6f986332989bc7ea465794f1c2b6856"
        network_throughput_bytes_per_sec: 292016122948
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "d1dbd5305c09a5bbfd5a1e9c7c51dd3f"
        network_throughput_bytes_per_sec: 342618151
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "f08943c7fb325a80c9aedfc3d9219cfb"
        network_throughput_bytes_per_sec: 1242435732
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "aa53481ae940f8be73d0809b81bd85ee"
        network_throughput_bytes_per_sec: 33755343806
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "6411603f6b84e86918c660203d2586d6"
        network_throughput_bytes_per_sec: 182449888
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "6823bce11114d8ceb5386584c6072fc8"
        network_throughput_bytes_per_sec: 86122792
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "e70429d2d9bdfaadf7c6d6cd451ad2d0"
        network_throughput_bytes_per_sec: 174624829
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "ea38ae3f2a296149dde0cac2c673fd8f"
        network_throughput_bytes_per_sec: 263586259582
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "61c2170cfeaf7234b58eea7d3c5cc136"
        network_throughput_bytes_per_sec: 15856283078
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "f57f58c135e77c22b55a8bdbe156e2a4"
        network_throughput_bytes_per_sec: 17673026360
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "abd8c8f7ce2fd95b63645316537b74e7"
        network_throughput_bytes_per_sec: 289717760542
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "47bea0ecc1ea80cb91e9b291e81894f8"
        network_throughput_bytes_per_sec: 65284552465
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "275f04921ccc4b0fa29127fb3a80a2f2"
        network_throughput_bytes_per_sec: 4457321635
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "f6d54f4055232256fcd6c0bf18d19f73"
        network_throughput_bytes_per_sec: 142461701757
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "44d6bfd785c449cc8640234e220f77aa"
        network_throughput_bytes_per_sec: 12531982025
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "4ffa457833b6c4ff6e5147781e31302a"
        network_throughput_bytes_per_sec: 103156802223
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "d9af6aa7b045746dbcb1015482010070"
        network_throughput_bytes_per_sec: 29791629968
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "c5eb925239f355be19b2a2ae51019597"
        network_throughput_bytes_per_sec: 206503744116
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "d28e01182634547a1027d109446f3b88"
        network_throughput_bytes_per_sec: 6836280185
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "a69547b3c276d3341642608b1db494a6"
        network_throughput_bytes_per_sec: 90527151860
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "83591db4b7a2726af43dd49fc404007b"
        network_throughput_bytes_per_sec: 112599898656
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "15185f0d28c7a7edc1b4aecfa6d8f221"
        network_throughput_bytes_per_sec: 249948514054
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "99043b67c066b466fdfabf4ba0e10d9d"
        network_throughput_bytes_per_sec: 334503879
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "4a5e09d90eb94f08a4eb31421f1e8443"
        network_throughput_bytes_per_sec: 76497126892
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "561c9e59f33d316b74205823cb42d04d"
        network_throughput_bytes_per_sec: 285606702146
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2097152
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "b01e85aab5ef3da6357ca6f9cfb67b8f"
        network_throughput_bytes_per_sec: 96956830291
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "bee86716cff01212fb687be758fc96e1"
        network_throughput_bytes_per_sec: 271017233924
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 512
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "a9dda3180d9de81b8ac47f5af4e3717c"
        network_throughput_bytes_per_sec: 171596145
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 131072
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "aedd2df037a65be1250f90a94291ed61"
        network_throughput_bytes_per_sec: 13073861652
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "e824c0c6cda89c864c487464c4714920"
        network_throughput_bytes_per_sec: 115599104957
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "31a79cde8d9fdcd983bb9252c3c92f73"
        network_throughput_bytes_per_sec: 46920350814
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "a0c14787b94ec324f39d4b4cde6aaee8"
        network_throughput_bytes_per_sec: 2500419687
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16777216
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "21ea7fae57499ff2b1700818db1ccc37"
        network_throughput_bytes_per_sec: 117049187304
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "a5c25d90d3703c3e05a5428a5fbafe10"
        network_throughput_bytes_per_sec: 2575290789
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "353ed7150bd91617fa8843f2620c704d"
        network_throughput_bytes_per_sec: 87089641
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "0cb65c209a26b3c7c2d519854b62c67c"
        network_throughput_bytes_per_sec: 9752380952
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "7d42a2c52405b0487725b632c32b3246"
        network_throughput_bytes_per_sec: 57640194044
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "d1ed25619ddb7091e280b64f8bf17b71"
        network_throughput_bytes_per_sec: 640300140
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 33554432
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "71a05d2710cbb210f44bd79ab02e4544"
        network_throughput_bytes_per_sec: 144412841361
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1024
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "3c0dab67f668aa96622d925405aa4c35"
        network_throughput_bytes_per_sec: 352404714
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 262144
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "1c34dad9da64a4a58f0c8f86c14a73a2"
        network_throughput_bytes_per_sec: 32431522949
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "260f965f1a4678225622c8bb1bb605bd"
        network_throughput_bytes_per_sec: 109586962343
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "0f9cdfcb5a2647c85e89d23875e02e61"
        network_throughput_bytes_per_sec: 90820399
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 268435456
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "30228c58f0f8bfd498c30a0b4c75491e"
        network_throughput_bytes_per_sec: 290438102739
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 67108864
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "96c966eff2db96826fa05a63d70abdfb"
        network_throughput_bytes_per_sec: 244936493728
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "b9f629195d1e0d1afeebe40aef9955df"
        network_throughput_bytes_per_sec: 77324336780
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "9ca39fa8794daa9631163ee0e266097e"
        network_throughput_bytes_per_sec: 668352778
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "393073cf2c2fb7f64008edb89f2f877a"
        network_throughput_bytes_per_sec: 55877860968
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "1f4b3db733dab96bb8fa97d8f2bb2c7e"
        network_throughput_bytes_per_sec: 292410046019
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "faccbf0108d642668cb20ee319a39541"
        network_throughput_bytes_per_sec: 286685266708
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8192
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "dd72a6e820b6d534b73545dc695b9277"
        network_throughput_bytes_per_sec: 2518484359
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4194304
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "8c4f72b22cf1c427b5192fde2275b82d"
        network_throughput_bytes_per_sec: 70532512139
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "067cb94ec99578ac4a5d635f77d6836e"
        network_throughput_bytes_per_sec: 1329007138
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 32768
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "9ea76d324a2a87d6007857f1bbd58e8b"
        network_throughput_bytes_per_sec: 7858504706
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "9efe18a44677d5ebf6d950d12f0105d0"
        network_throughput_bytes_per_sec: 111468000783
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 2048
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "baf06cabffc7267e4ba88f3f0469f867"
        network_throughput_bytes_per_sec: 657252888
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 65536
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "1f93b4ce5f502f4229278ea4c2936bb8"
        network_throughput_bytes_per_sec: 11199384799
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 1048576
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "04a5265203d9bcb5e98c819340f31d6c"
        network_throughput_bytes_per_sec: 69698295057
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 8388608
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "e78c4bfa48ee5d7f81743a050aa5f803"
        network_throughput_bytes_per_sec: 74252717998
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 536870912
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "7c2845488c50871c6df2b01b09b93607"
        network_throughput_bytes_per_sec: 271990404586
      }
      entries {
        instruction {
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 524288
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 target: 2 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 4 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 6 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "3aa698dd32575d30602a31ddba9fefac"
        network_throughput_bytes_per_sec: 41157750127
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "87766ec8522f636b65d29ffbbac0f005"
        network_throughput_bytes_per_sec: 4868583314
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 16384
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "0a1bea9a023ace07a93df0c757a64d8d"
        network_throughput_bytes_per_sec: 4709737693
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 4096
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 1 }
          source_target_pairs { source: 1 }
          source_target_pairs { source: 2 target: 3 }
          source_target_pairs { source: 3 target: 2 }
          source_target_pairs { source: 4 target: 5 }
          source_target_pairs { source: 5 target: 4 }
          source_target_pairs { source: 6 target: 7 }
          source_target_pairs { source: 7 target: 6 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "923306d188529fd23828978fba917eca"
        network_throughput_bytes_per_sec: 1285019607
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
          name: "collective-permute"
          opcode: "collective-permute"
          shape {
            element_type: F32
            dimensions: 134217728
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
          metadata {}
          channel_id: 1
          id: 4294967299
          operand_ids: 4294967298
          source_target_pairs { target: 4 }
          source_target_pairs { source: 1 target: 5 }
          source_target_pairs { source: 2 target: 6 }
          source_target_pairs { source: 3 target: 7 }
          frontend_attributes {}
          statistics_viz {}
        }
        fingerprint: "00842feada0344771c1b4e414c197917"
        network_throughput_bytes_per_sec: 269253988123
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
    }
  }
)pb";
// END_DEFAULT_PERF_TABLE

#endif  // XLA_SERVICE_GPU_MODEL_COLLECTIVE_INTERPOLATOR_DATA_H_
