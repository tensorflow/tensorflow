

// void ACCNAME::Output_Handler() {
// #pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable
// = \
//     outS

//   outS.write(0);
//   DATA last = {5000, 1};
//   wait();
//   while (1) {
//     outS.write(1);
//     DATA data0 = vars[0].out_fifo.read();
//     wait();
//     outS.write(2);
//     DATA data1 = vars[1].out_fifo.read();
//     wait();
//     outS.write(3);
//     DATA data2 = vars[2].out_fifo.read();
//     wait();
//     outS.write(4);
//     if (!data0.tlast) dout1.write(data0);
//     if (!data1.tlast) dout1.write(data1);
//     dout1.write(data2);

//     outS.write(5);
//     wait();
//   }
// }

// void ACCNAME::Output_Handler() {
// #pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable
// = \
//     outS

//   outS.write(0);
//   DATA last = {5000, 1};
//   wait();
//   while (1) {
//     outS.write(1);
//     DATA data0 = vars[0].out_fifo.read();
//     DATA data1 = vars[1].out_fifo.read();
//     DATA data2 = vars[2].out_fifo.read();
//     wait();
//     outS.write(2);
//     if (!data0.tlast) dout1.write(data0);
//     if (!data1.tlast) dout1.write(data1);
//     dout1.write(data2);

//     outS.write(5);
//     wait();
//   }
// }

// void ACCNAME::Output_Handler() {
// #pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable
// = \
//     outS

//   outS.write(0);
//   DATA last = {5000, 1};
//   wait();
//   while (1) {
//     outS.write(1);
//     DATA data0 = vars.vars_0.out_fifo.read();
//     DATA data1 = vars.vars_1.out_fifo.read();
//     DATA data2 = vars.vars_2.out_fifo.read();
//     wait();
//     outS.write(2);
//     if (!data2.tlast) dout1.write(data0);
//     if (!data2.tlast) dout1.write(data1);
//     dout1.write(data2);

//     outS.write(5);
//     wait();
//   }
// }

// void ACCNAME::Output_Handler() {
// #pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable
// = \
//     outS

//   outS.write(0);
//   DATA last = {5000, 1};
//   wait();
//   while (1) {
//     outS.write(1);
//     wait();
//     DATA data0 = vars.vars_0.out_fifo.read();
//     DATA data1 = vars.vars_1.out_fifo.read();
//     DATA data2 = vars.vars_2.out_fifo.read();
//     DATA data3 = vars.vars_3.out_fifo.read();
//     DATA data4 = vars.vars_4.out_fifo.read();
//     DATA data5 = vars.vars_5.out_fifo.read();
//     DATA data6 = vars.vars_6.out_fifo.read();
//     DATA data7 = vars.vars_7.out_fifo.read();

//     wait();
//     outS.write(2);
//     if (!data0.tlast) dout1.write(data0);
//     if (!data1.tlast) dout1.write(data1);
//     if (!data2.tlast) dout1.write(data2);
//     if (!data3.tlast) dout1.write(data3);
//     if (!data4.tlast) dout1.write(data4);
//     if (!data5.tlast) dout1.write(data5);
//     if (!data6.tlast) dout1.write(data6);
//     dout1.write(data7);

//     outS.write(5);
//     wait();
//   }
// }

// void ACCNAME::Output_Handler() {
// #pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable
// = \
//     outS

//   outS.write(0);
//   DATA last = {5000, 1};
//   DATA data[PE_COUNT];
//   for (int i = 0; i < PE_COUNT; i++)
//     data[i].tlast = 0;

//   out_done.write(false);
//   wait();
//   while (1) {
//     outS.write(1);
//     while (!out_fifo_filled())
//       DWAIT();

//     for (int i = 0; i < PE_COUNT; i++) {
// #pragma HLS unroll
//       data[i].data = vars.get(i);
//     }

//     wait();
//     outS.write(2);

//     for (int i = 0; i < PE_COUNT; i++) {
// #pragma HLS unroll
//       dout1.write(data[i]);
//     }

//     if (store_done() && !out_fifo_filled()) {
//       dout1.write(last);
//       out_done.write(true);
//       while (out_done_start)
//         DWAIT();
//       out_done.write(false);
//     }

//     outS.write(5);
//     wait();
//   }
// }

// void ACCNAME::Output_Handler() {
// #pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable
// = \
//     outS

//   outS.write(0);
//   int data[PE_COUNT];
//   bool tlast[PE_COUNT];
//   for (int i = 0; i < PE_COUNT; i++)
//     tlast[i] = 0;

//   wait();
//   while (1) {
//     outS.write(1);
//     wait();
//     for (int i = 0; i < PE_COUNT; i++) {
// #pragma HLS unroll
//       DATA d = vars.get(i);
//       data[i] = d.data;
//       tlast[i] = d.tlast;
//     }

//     wait();
//     outS.write(2);

//     for (int i = 0; i < PE_COUNT - 1; i++) {
// #pragma HLS unroll
//       DATA d = {data[i], tlast[i]};
//       if (!tlast[i]) dout1.write(d);
//     }
//     DATA d = {data[PE_COUNT - 1], tlast[PE_COUNT - 1]};
//     dout1.write(d);
//     outS.write(5);
//     DWAIT();
//   }
// }

void ACCNAME::Output_Handler() {
#pragma HLS resource core = AXI4LiteS metadata = "-bus_bundle slv0" variable = \
    outS

  outS.write(0);
  int data[PE_COUNT];
  bool tlast = false;
  DATA last = {5000, 1};
  wait();
  while (1) {
    outS.write(1);
    tlast = false;
    wait();
    for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
      DATA d = vars.get(i);
      data[i] = d.data;
      tlast = tlast || d.tlast;
    }

    wait();
    outS.write(2);

    for (int i = 0; i < PE_COUNT; i++) {
#pragma HLS unroll
      DATA d;
      d.data = data[i];
      d.tlast = 0;
      dout1.write(d);
    }
    if (tlast) {
      dout1.write(last);
    }
    outS.write(5);
    DWAIT();
  }
}