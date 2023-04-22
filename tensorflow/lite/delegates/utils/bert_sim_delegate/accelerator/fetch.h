#include "acc.h"

void ACCNAME::fetch() {
  int start_count = 0;
  int done_count = 0;

  crf_val = 0;
  crx_val = 0;
  ra_val = 0;

  opcode insn(0, 0);
  wait();
  while (true) {
    wait();

    if (start_acc.read() > start_count) {
      start_count++;
      done_count++;

      crf_val = crf.read();
      crx_val = crx.read();
      ra_val = ra.read();

      if (crx_val > 0) {
        pl = crx_val;
        pr = 0;
        msk = 0;
        sm = 0;
      } else {
        pl = 1;
        pr = -crx_val;
        msk = (1 << -crx_val) - 1;
        sm = msk >> 1;
      }

      int insn_len = insn_count.read();
      int insn_idx = 0;
      ins_count->value +=insn_len;
      for (int pc = 0; pc < insn_len; pc++) {
        sc_uint<64> in2 = insn_port.read(insn_addr + insn_idx);
        sc_uint<64> in1 = insn_port.read(insn_addr + insn_idx + 1);
        insn_idx += 2;
        opcode insn_read(in1, in2);
        unsigned long long p1 = in1;
        unsigned long long p2 = in2;
        insn = insn_read;

        if (insn.op == 1) {
          wgt_insn1 = in1;
          wgt_insn2 = in2;
          loading.write(true);
          wgt_load.write(true);
          DWAIT();
          while (wgt_load.read()) wait();
          loading.write(false);
          DWAIT();

        } else if (insn.op == 2) {
          inp_insn1 = in1;
          inp_insn2 = in2;
          loading.write(true);
          inp_load.write(true);
          DWAIT();
          while (inp_load.read()) wait();
          loading.write(false);
          DWAIT();
        } else if (insn.op == 3) {
          bias_insn1 = in1;
          bias_insn2 = in2;
          loading.write(true);
          bias_load.write(true);
          DWAIT();
          while (bias_load.read()) wait();
          loading.write(false);
          DWAIT();
        } else {
          store_doffset.write(insn.doffset);
          store_dstride.write(insn.dstride);
          inp_block.write(insn.y_size);
          wgt_block.write(insn.x_size);

          depth_val.write(depth.read());
          schedule.write(true);

          DWAIT();
          while (schedule.read()) wait();
        }
      }
    }

    DWAIT();
    sc_pause();
    wait();
  }
}