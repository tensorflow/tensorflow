void ACCNAME::Worker1(){
	ACC_DTYPE<32> out[16][4];
#pragma HLS array_partition variable=out complete dim=0

	w1S.write(0);
	wait();
	while(1){

		while(gemm_unit_1_ready.read()) {
			w1S.write(10);DWAIT();
		}
#ifndef __SYNTHESIS__
		w1S.write(2);DWAIT();
#endif
		int d = (depth/4);
		int l_pointer=gemm_unit_1_l_pointer.read();
		gemm_unit_1_iwuse.write(1);
		gemm_unit_1_ready.write(1);
		w1S.write(3);
		wait();
		VM_PE(lhsdata1a,lhsdata2a,lhsdata3a,lhsdata4a,
			  rhs1a_1,rhs1b_1,rhs1c_1,rhs1d_1,
			  out,d,l_pointer,0);
		w1S.write(4);
		gemm_unit_1_iwuse.write(0);

		while(write1.read()){
			w1S.write(9);DWAIT();
		}

		for(int i=0;i<16;i++){
#pragma HLS unroll
			g1[i] = out[i][0];
		}
		write1.write(1);

#ifndef __SYNTHESIS__
		gouts->array[0]+=16;DWAIT(2);
#endif
	}
}

// void ACCNAME::Worker2(){
// 	ACC_DTYPE<32> out[16][4];
// #pragma HLS array_partition variable=out complete dim=0

// 	w2S.write(0);
// 	wait();
// 	while(1){

// 		while(gemm_unit_2_ready.read()) {
// 			w2S.write(10);DWAIT();
// 		}
// #ifndef __SYNTHESIS__
// 		w2S.write(2);DWAIT();
// #endif
// 		int d = (depth/4);
// 		int l_pointer=gemm_unit_2_l_pointer.read();
// 		gemm_unit_2_iwuse.write(1);
// 		gemm_unit_2_ready.write(1);
// 		w2S.write(3);
// 		wait();
// 		VM_PE(lhsdata1b,lhsdata2b,lhsdata3b,lhsdata4b,
// 			  rhs2a_1,rhs2b_1,rhs2c_1,rhs2d_1,
// 			  out,d,l_pointer,1);
// 		w2S.write(4);
// 		gemm_unit_2_iwuse.write(0);

// 		while(write2.read()){
// 			w2S.write(9);DWAIT();
// 		}

// 		for(int i=0;i<16;i++){
// #pragma HLS unroll
// 			g2[i] = out[i][0];
// 		}
// 		write2.write(1);

// #ifndef __SYNTHESIS__
// 		gouts->array[1]+=16;DWAIT(2);
// #endif
// 	}
// }

// void ACCNAME::Worker3(){
// 	ACC_DTYPE<32> out[16][4];
// #pragma HLS array_partition variable=out complete dim=0

// 	w3S.write(0);
// 	wait();
// 	while(1){

// 		while(gemm_unit_3_ready.read()) {
// 			w3S.write(10);
// 			DWAIT();
// 		}
// #ifndef __SYNTHESIS__
// 		w3S.write(2);DWAIT();
// #endif
// 		int d = (depth/4);
// 		int l_pointer=gemm_unit_3_l_pointer.read();
// 		gemm_unit_3_iwuse.write(1);
// 		gemm_unit_3_ready.write(1);
// 		w3S.write(3);
// 		wait();
// 		VM_PE(lhsdata1c,lhsdata2c,lhsdata3c,lhsdata4c,
// 			  rhs3a_1,rhs3b_1,rhs3c_1,rhs3d_1,
// 			  out,d,l_pointer,2);
// 		w3S.write(4);
// 		gemm_unit_3_iwuse.write(0);

// 		while(write3.read()){
// 			w3S.write(9);DWAIT();
// 		}

// 		for(int i=0;i<16;i++){
// #pragma HLS unroll
// 			g3[i] = out[i][0];
// 		}
// 		write3.write(1);

// #ifndef __SYNTHESIS__
// 		gouts->array[2]+=16;DWAIT(2);
// #endif
// 	}
// }

// void ACCNAME::Worker4(){
// 	ACC_DTYPE<32> out[16][4];
// #pragma HLS array_partition variable=out complete dim=0

// 	w4S.write(0);
// 	wait();
// 	while(1){

// 		while(gemm_unit_4_ready.read()) {
// 			w4S.write(10);
// 			DWAIT();
// 		}
// #ifndef __SYNTHESIS__
// 		w4S.write(2);DWAIT();
// #endif
// 		int d = (depth/4);
// 		int l_pointer = gemm_unit_4_l_pointer.read();
// 		gemm_unit_4_iwuse.write(1);
// 		gemm_unit_4_ready.write(1);
// 		w4S.write(3);
// 		wait();
// 		VM_PE(lhsdata1d,lhsdata2d,lhsdata3d,lhsdata4d,
// 			  rhs4a_1,rhs4b_1,rhs4c_1,rhs4d_1,
// 			  out,d,l_pointer,3);
// 		w4S.write(4);
// 		gemm_unit_4_iwuse.write(0);

// 		while(write4.read()){
// 			w4S.write(9);DWAIT();
// 		}

// 		for(int i=0;i<16;i++){
// #pragma HLS unroll
// 			g4[i] = out[i][0];
// 		}
// 		write4.write(1);

// #ifndef __SYNTHESIS__
// 		gouts->array[3]+=16;DWAIT(2);
// #endif
// 	}
// }


