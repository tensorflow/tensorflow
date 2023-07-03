#define COMPUTE -1
#define OUTPUT -2
#include <iomanip>

void ACCNAME::Input_Handler() {
  // clang-format off
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=scheduleS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=tempS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=on
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=inS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_0.computeS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_0.sendS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_1.computeS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_1.sendS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_2.computeS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_2.sendS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_3.computeS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_3.sendS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_4.computeS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_4.sendS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_5.computeS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_5.sendS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_6.computeS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_6.sendS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_7.computeS
#pragma HLS resource core=AXI4LiteS metadata="-bus_bundle slv0" variable=vars.vars_7.sendS
  // clang-format on
  DATA d;

  load_wgt.write(false);
  load_inp.write(false);
  load_map.write(false);
  load_data.write(false);

#ifdef __SYNTHESIS__
  vars.vars_0.computeS.write(0);
  vars.vars_0.sendS.write(0);
  vars.vars_1.computeS.write(0);
  vars.vars_1.sendS.write(0);
  vars.vars_2.computeS.write(0);
  vars.vars_2.sendS.write(0);
  vars.vars_3.computeS.write(0);
  vars.vars_3.sendS.write(0);
  vars.vars_4.computeS.write(0);
  vars.vars_4.sendS.write(0);
  vars.vars_5.computeS.write(0);
  vars.vars_5.sendS.write(0);
  vars.vars_6.computeS.write(0);
  vars.vars_6.sendS.write(0);
  vars.vars_7.computeS.write(0);
  vars.vars_7.sendS.write(0);
#endif

  on.write(true);
  inS.write(0);
  wait();
  while (1) {

    inS.write(1);
    wait();

    opcode op = opcode(din1.read().data.to_uint());
    if (op.load_con) {
      inS.write(2);
      wait();
      depth = din1.read().data;
      row_size = din1.read().data;
      cols_per_filter = din1.read().data;
      inp_rows = din1.read().data;
      ra = din1.read().data;
      for (int i = 0; i < PE_COUNT; i++) {
        pe_cols[i] = cols_per_filter * i;
      }
    }

    if (op.load_wgt || op.load_inp || op.load_map || op.load_col_map) {
      if (op.load_wgt) load_wgt.write(true);
      if (op.load_inp) load_inp.write(true);
      if (op.load_map) load_map.write(true);
      if (op.load_col_map) load_col_map.write(true);

      load_data.write(true);
      inS.write(3);
      wait();
      while (load_data.read())
        wait();
      load_wgt.write(false);
      load_inp.write(false);
      load_map.write(false);
      load_col_map.write(false);
    }

    if (op.schedule) {
      schedule.write(true);
      inS.write(4);
      wait();
      while (schedule.read())
        wait();
    }
    DWAIT();
  }
}
