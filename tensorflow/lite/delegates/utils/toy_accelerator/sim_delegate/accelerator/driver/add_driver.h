#ifndef ADD_DRIVER
#define ADD_DRIVER

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
namespace tflite_toysim {

void BlockAdd(acc_container& drv) {
  int i_len = 0;
  int* DMA_input_buffer = drv.sdma->dma_get_inbuffer();
  DMA_input_buffer[i_len++] = drv.length / 4;
  DMA_input_buffer[i_len++] = drv.lshift;
  DMA_input_buffer[i_len++] = drv.in1_off;
  DMA_input_buffer[i_len++] = drv.in1_sv;
  DMA_input_buffer[i_len++] = drv.in1_mul;
  DMA_input_buffer[i_len++] = drv.in2_off;
  DMA_input_buffer[i_len++] = drv.in2_sv;
  DMA_input_buffer[i_len++] = drv.in2_mul;
  DMA_input_buffer[i_len++] = drv.out1_off;
  DMA_input_buffer[i_len++] = drv.out1_sv;
  DMA_input_buffer[i_len++] = drv.out1_mul;
  DMA_input_buffer[i_len++] = drv.qa_max;
  DMA_input_buffer[i_len++] = drv.qa_min;
  int8_t a_val[4];
  int8_t b_val[4];
  int* aval = reinterpret_cast<int*>(a_val);
  int* bval = reinterpret_cast<int*>(b_val);

  for (int i = 0; i < drv.length; i += 4) {
    a_val[0] = drv.input_A[i + 0];
    a_val[1] = drv.input_A[i + 1];
    a_val[2] = drv.input_A[i + 2];
    a_val[3] = drv.input_A[i + 3];
    b_val[0] = drv.input_B[i + 0];
    b_val[1] = drv.input_B[i + 1];
    b_val[2] = drv.input_B[i + 2];
    b_val[3] = drv.input_B[i + 3];
    DMA_input_buffer[i_len++] = aval[0];
    DMA_input_buffer[i_len++] = bval[0];
  }

  drv.sdma->dma_start_send(i_len);
  drv.sdma->dma_wait_send();
  drv.sdma->dma_start_recv(drv.length / 4);
  drv.sdma->dma_wait_recv();

  drv.profile->saveProfile(drv.acc->profiling_vars);
  int8_t* oval = reinterpret_cast<int8_t*>(drv.sdma->dma_get_outbuffer());
  for (int i = 0; i < drv.length; i++) {
    drv.output_C[i] = oval[i];
  }
}

void Entry(acc_container& drv) {
#ifdef DELEGATE_VERBOSE
  cout << "ToyAdd" << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "ADD Layer: " << drv.layer << endl;
  cout << "Input Length: " << drv.length << endl;
  cout << "===========================" << endl;
#endif
  BlockAdd(drv);
}

}  // namespace tflite_secda
#endif // ADD_DRIVER