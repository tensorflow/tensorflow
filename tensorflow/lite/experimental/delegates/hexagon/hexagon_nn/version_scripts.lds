VERS_1.0 {
  global:
    hexagon_nn_config;
    hexagon_nn_init;
    hexagon_nn_prepare;
    hexagon_nn_set_powersave_level;
    hexagon_nn_set_debug_level;
    hexagon_nn_append_node;
    hexagon_nn_append_const_node;
    hexagon_nn_execute;
    hexagon_nn_execute_new;
    hexagon_nn_teardown;
    hexagon_nn_snpprint;
    hexagon_nn_getlog;
    hexagon_nn_get_perfinfo;
    hexagon_nn_reset_perfinfo;
    hexagon_nn_op_id_to_name;
    hexagon_nn_global_teardown;
    hexagon_nn_global_init;
    hexagon_nn_is_device_supported;
    hexagon_nn_version;
    hexagon_nn_hexagon_interface_version;

  # Hide everything else.
  local:
    *;
};