#ifndef VM_TCONV_DRIVER
#define VM_TCONV_DRIVER

// #define SYSC

#include "acc_container.h"
#include "mm2im_driver.h"
#include "mm2im_util.h"
#include <iostream>
#include <utility>

// Pre-Defined Address for Accelerator
#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000
#define dma_in0 0x16000000
#define dma_in1 0x18000000
#define dma_in2 0x1a000000
#define dma_in3 0x1c000000
#define dma_out0 0x16800000
#define dma_out1 0x18800000
#define dma_out2 0x1a800000
#define dma_out3 0x1c800000
#define DMA_BL 4194304

using namespace std;

unsigned int dma_addrs[4] = {0, 0, 0, 0};
unsigned int dma_addrs_in[4] = {0, 0, 0, 0};
unsigned int dma_addrs_out[4] = {0, 0, 0, 0};

bool test_accelerator(int *params, acc_container &drv, int id) {
  int stride_x = params[0];
  int stride_y = params[1];
  int filters = params[2];
  int kernel_size = params[3];
  int in1 = params[4];
  int in2 = params[5];
  int in3 = params[6];
  string padding = params[7] == 1 ? "same" : "valid";
  bool verbose = params[8] == 1 ? true : false;

  int out1, out2, out3, rows, cols, depth;
  int pt, pb, pl, pr;
  calParams(stride_x, stride_y, filters, kernel_size, in1, in2, in3, padding,
            rows, cols, depth, out1, out2, out3, pt, pb, pl, pr);

  // Input Params
  int ra = 26;
  int rhs_offset = 0;
  int lhs_offset = 0;
  int output_height = out1;
  int output_width = out2;
  int output_depth = out3;
  int filter_height = kernel_size;
  int filter_width = kernel_size;
  int padding_top = pt;
  int padding_left = pl;
  int padding_bottom = pb;
  int padding_right = pr;
  int stride_height = stride_x;
  int stride_width = stride_y;
  int rounded_depth = roundUp(in3, 16);

  // Reference
  int8_t *ref_input = new int8_t[in3 * in1 * in2]();
  int8_t *ref_weight = new int8_t[in3 * filters * kernel_size * kernel_size]();
  int *ref_wt_sum = new int[filters * kernel_size * kernel_size]();
  int *ref_bias = new int[filters]();
  int *ref_crf = new int[filters]();
  int8_t *ref_crx = new int8_t[filters]();

  int *ref_dst = new int[in1 * in2 * filters * kernel_size * kernel_size]();
  int8_t *ref_padded_weights = new int8_t[rounded_depth * rows]();
  int8_t *ref_padded_inputs = new int8_t[rounded_depth * cols]();
  int32_t *ref_dex_map =
      new int[in1 * in2 * filters * kernel_size * kernel_size]();
  struct mm2im_params rmp(stride_x, stride_y, filters, kernel_size, in1, in2,
                          in3, rows, cols, depth, out1, out2, out3, padding_top,
                          padding_bottom, padding_left, padding_right, true);

  for (int i = 0; i < in3 * in1 * in2; i++)
    ref_input[i] = i % 128;

  for (int k = 0; k < kernel_size * kernel_size; k++) {
    for (int j = 0; j < filters; j++) {
      for (int i = 0; i < in3; i++) {
        ref_weight[k * filters * in3 + j * in3 + i] = j;
      }
    }
  }
  for (int j = 0; j < filters; j++) {
    ref_bias[j] = (j + 1) * 100;
    ref_crf[j] = (j + 1) * 100000000;
    ref_crx[j] = (j + 1) % 10;
  }

  preload_weights(ref_weight, depth, rows, ref_wt_sum, ref_padded_weights);
  preload_inputs(ref_input, depth, cols, ref_padded_inputs);
  col2im_mapping_v2(rmp.o3, rmp.o1, rmp.o2, rmp.ks, rmp.ks, rmp.pt, rmp.pl,
                    rmp.pb, rmp.pr, rmp.sx, rmp.sy, ref_dex_map);
  rmp.input_data = ref_input;
  rmp.weight_data = ref_weight;
  rmp.output_data = ref_dst;
  rmp.dex_map = ref_dex_map;
  rmp.wt_sum = ref_wt_sum;
  rmp.bias_data = ref_bias;
  rmp.crf = ref_crf;
  rmp.crx = ref_crx;
  rmp.ra = ra;
  rmp.MM2IM_compute();

  // Accelerator
  int8_t *acc_input = new int8_t[in3 * in1 * in2]();
  int8_t *acc_weight = new int8_t[in3 * filters * kernel_size * kernel_size]();
  int *acc_wt_sum = new int[filters * kernel_size * kernel_size]();
  int *acc_dst = new int[in1 * in2 * filters * kernel_size * kernel_size]();
  int8_t *acc_padded_weights = new int8_t[rounded_depth * rows]();
  int8_t *acc_padded_inputs = new int8_t[rounded_depth * cols]();
  int32_t *acc_dex_map =
      new int[in1 * in2 * filters * kernel_size * kernel_size]();
  struct mm2im_params amp(stride_x, stride_y, filters, kernel_size, in1, in2,
                          in3, rows, cols, depth, out1, out2, out3, padding_top,
                          padding_bottom, padding_left, padding_right, true);
  for (int i = 0; i < in3 * in1 * in2; i++)
    acc_input[i] = i % 128;

  for (int j = 0; j < filters; j++) {
    for (int k = 0; k < kernel_size * kernel_size; k++) {
      for (int i = 0; i < in3; i++) {
        acc_weight[j * kernel_size * kernel_size * in3 + k * in3 + i] = j;
      }
    }
  }
  preload_weights(acc_weight, depth, rows, acc_wt_sum, acc_padded_weights);
  preload_inputs(acc_input, depth, cols, acc_padded_inputs);
  int32_t *acc_loaded_wgts = reinterpret_cast<int32_t *>(acc_padded_weights);
  int32_t *acc_loaded_inps = reinterpret_cast<int32_t *>(acc_padded_inputs);
  col2im_mapping_v3(amp.o3, amp.o1, amp.o2, amp.ks, amp.ks, amp.pt, amp.pl,
                    amp.pb, amp.pr, amp.sx, amp.sy, acc_dex_map);
  amp.input_data = acc_input;
  amp.weight_data = acc_weight;
  amp.output_data = acc_dst;
  amp.dex_map = acc_dex_map;
  amp.wt_sum = acc_wt_sum;
  amp.bias_data = ref_bias;
  amp.crf = ref_crf;
  amp.crx = ref_crx;
  amp.ra = ra;
  amp.compute = verbose;
  amp.MM2IM_o1_map();

  drv.lhs_offset = 0;
  drv.ih = in1;
  drv.iw = in2;
  drv.ic = in3;
  drv.f = filters;
  drv.ks = kernel_size;
  drv.o1 = out1;
  drv.o2 = out2;
  drv.o3 = out3;
  drv.sx = stride_x;
  drv.sy = stride_y;
  drv.pt = padding_top;
  drv.pl = padding_left;
  drv.pb = padding_bottom;
  drv.pr = padding_right;
  drv.rows = rows;
  drv.cols = cols;
  drv.depth = depth;
  drv.loaded_weights = acc_loaded_wgts;
  drv.loaded_inputs = acc_loaded_inps;
  drv.weights = acc_weight;
  drv.inputs = acc_input;
  drv.bias = ref_bias;
  drv.crf = ref_crf;
  drv.crx = ref_crx;
  drv.ra = ra;

  drv.o1_lengths = &amp.o1_lengths[0];
  drv.o1_starts = &amp.o1_starts[0];
  drv.o1_ends = &amp.o1_ends[0];

  drv.col_dexs = amp.col_dexs;
  drv.out_dexs = amp.out_dexs;

  drv.mm2im_map = amp.mm2im_map;
  drv.o1_map = amp.o1_map;
  drv.output_data = acc_dst;

  drv.wt_sum = acc_wt_sum;

  drv.verb = verbose;
  mm2im_driver::Entry(drv);

  // Compare
  for (int i = 0; i < in1 * in2 * filters * kernel_size * kernel_size; i++) {
    if (acc_dst[i] != ref_dst[i]) {
      std::cout << "Problem id: " << id << std::endl;
      std::cout << "Mismatch at " << i << " ref: " << ref_dst[i]
                << " acc: " << acc_dst[i] << std::endl;
      std::cout << "Test " << id << " FAILED" << std::endl;
      return 1;
    }
  }
  std::cout << "Test " << id << " PASSED" << std::endl;
  return 0;
}

int main() {

  struct acc_container drv;
#ifdef SYSC
  static struct sysC_sigs scs1(1);
  static ACCNAME acc("MM2IM");
  static struct multi_dma mdma(4, dma_addrs, dma_addrs_in, dma_addrs_out,
                               563840);
  static struct Profile profile;
  sysC_init();
  sysC_binder(&acc, &mdma, &scs1);
#else
  int *accelerator = getAccBaseAddress<int>(acc_address, 65536);
  static unsigned int dma_addrs[4] = {dma_addr0, dma_addr1, dma_addr2,
                                      dma_addr3};
  static unsigned int dma_addrs_in[4] = {dma_in0, dma_in1, dma_in2, dma_in3};
  static unsigned int dma_addrs_out[4] = {dma_out0, dma_out1, dma_out2,
                                          dma_out3};
  static struct multi_dma mdma(4, dma_addrs, dma_addrs_in, dma_addrs_out,
                               DMA_BL);
  drv.acc = accelerator;
#endif

  drv.mdma = &mdma;
  // drv.profile = &profile;

  // int params[8] = {1, 1, 128, 5, 7, 7, 256, 1};
  // int params[8] = {1, 1, 2, 3, 2, 2, 2, 1};
  // int params[8] = {1, 1, 2, 3, 2, 2, 4, 1};
  // int params[8] = {2, 2, 2, 3, 4, 4, 2, 1};

  // clang-format off
  int params[5][9] = {
  //  {2, 2, 2, 3, 4, 4, 2, 1,1}
  {1, 1, 9, 3, 4, 4, 2, 1,0}
  ,{1, 1, 2, 3, 2, 2, 2, 1,0}
  ,{1, 1, 2, 3, 2, 2, 4, 1,0}
  ,{2, 2, 2, 3, 4, 4, 2, 1,0}
  ,{1, 1, 128, 5, 7, 7, 256, 1,0}
  };
  // clang-format on

  for (int i = 0; i < 1; i++) {
    int ret = test_accelerator(params[i], drv, i);
    if (ret != 0)
      return ret;
  }

  return 0;
}

#endif // VM_TCONV_DRIVER