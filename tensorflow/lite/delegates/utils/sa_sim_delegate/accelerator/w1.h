void ACCNAME::Worker1(){
	ACC_DTYPE<8> in[16][16];
	ACC_DTYPE<8> we[16][16];
	ACC_DTYPE<32> prod[256];
	ACC_DTYPE<32> od[256];

#pragma HLS array_partition variable=od complete dim=0
#pragma HLS array_partition variable=prod complete dim=0
#pragma HLS array_partition variable=in complete dim=0
#pragma HLS array_partition variable=we complete dim=0

	w1S.write(0);
	wait();
	while(1){

		while(gemm_unit_1_ready.read()) {
			w1S.write(10);
			DWAIT();
		}

		int d = depth+30;
		w1S.write(1);
		wait();

		for(int i=0;i<256;i++){
#pragma HLS unroll
			od[i]=0;
		}

		for(int i=0;i<d;i++){
			for(int i=15;i>0;i--){
#pragma HLS unroll
				for(int j=0;j<16;j++){
#pragma HLS unroll
					in[j][i] = in[j][i-1];
					we[i][j] = we[i-1][j];
				}
			}

			in[0][0] = sIs1.read();
			in[1][0] = sIs2.read();
			in[2][0] = sIs3.read();
			in[3][0] = sIs4.read();
			in[4][0] = sIs5.read();
			in[5][0] = sIs6.read();
			in[6][0] = sIs7.read();
			in[7][0] = sIs8.read();
			in[8][0] = sIs9.read();
			in[9][0] = sIs10.read();
			in[10][0] = sIs11.read();
			in[11][0] = sIs12.read();
			in[12][0] = sIs13.read();
			in[13][0] = sIs14.read();
			in[14][0] = sIs15.read();
			in[15][0] = sIs16.read();

			we[0][0] = sWs1.read();
			we[0][1] = sWs2.read();
			we[0][2] = sWs3.read();
			we[0][3] = sWs4.read();
			we[0][4] = sWs5.read();
			we[0][5] = sWs6.read();
			we[0][6] = sWs7.read();
			we[0][7] = sWs8.read();
			we[0][8] = sWs9.read();
			we[0][9] = sWs10.read();
			we[0][10] = sWs11.read();
			we[0][11] = sWs12.read();
			we[0][12] = sWs13.read();
			we[0][13] = sWs14.read();
			we[0][14] = sWs15.read();
			we[0][15] = sWs16.read();

			for(int i=0;i<10;i++){
#pragma HLS unroll
				for(int j=0;j<16;j++){
#pragma HLS unroll
					od[(i*16) + j] += in[j][i] * we[j][i];
					// if((i*16) + j == 0)cout << in[j][i] << " , " << we[j][i] << endl;
				}
			}
			for(int i=10;i<16;i++){
#pragma HLS unroll
				for(int j=0;j<16;j++){
#pragma HLS unroll
					prod[(i*16) + j] = mul_u8(in[j][(i)],we[j][i]);
					// if((i*16) + j == 0)cout << in[j][i] << " , " << we[j][i] << endl;
					od[(i*16) + j] += prod[(i*16) + j];
				}
			}

#ifndef __SYNTHESIS__
			g1_macs+=256;
			DWAIT(3);
#endif
		}

		
		w1S.write(4);
		DWAIT(8);
		while(write1.read()){
			w1S.write(9);
			DWAIT();
		}

		for(int i=0;i<256;i++){
#pragma HLS unroll
			g1[i]=od[i];
		}

		wait();
		write1.write(1);
		w1S.write(5);
		gemm_unit_1_ready.write(1);
		wait();

#ifndef __SYNTHESIS__
		gmacs->value+=256;
#endif
	}
}
