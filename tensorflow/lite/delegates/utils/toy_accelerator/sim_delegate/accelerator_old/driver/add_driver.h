#ifndef ACC_DRIVER
#define ACC_DRIVER

#include "acc_container.h"

namespace tflite_secda {

void BlockAdd(acc_container& drv) {
  drv.sdma->dmad->DMA_input_buffer[0] = 4;
  drv.sdma->dma_start_send(1);
  drv.sdma->dma_wait_send();
  int k = 0;
}

void Entry(acc_container& drv) {
  cout << "ADD - Layer: " << drv.layer << endl;
  cout << "===========================" << endl;
  //   cout << "Pre-ACC Info" << endl;
  //   cout << "padded_K: " << drv.pK << " K: " << drv.K << endl;
  //   cout << "padded_M: " << drv.pM << " M: " << drv.M << endl;
  //   cout << "padded_N: " << drv.pN << " N: " << drv.N << endl;
  cout << "===========================" << endl;

  BlockAdd(drv);
}

}  // namespace tflite_secda
#endif