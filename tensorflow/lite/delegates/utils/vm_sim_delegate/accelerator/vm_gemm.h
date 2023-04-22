void ACCNAME::VM_PE(
		ACC_DTYPE<32>* l1,ACC_DTYPE<32>* l2,ACC_DTYPE<32>* l3,ACC_DTYPE<32>* l4,
		ACC_DTYPE<32>* r1,ACC_DTYPE<32>* r2,ACC_DTYPE<32>* r3,ACC_DTYPE<32>* r4,
		ACC_DTYPE<32> out[][4],int d, int l_pointer, int wID){
	ACC_DTYPE<32> lhs_read[4];
	ACC_DTYPE<32> rhs_read[4];
	ACC_DTYPE<8> in_a[8];
	ACC_DTYPE<8> in_b[8];
	ACC_DTYPE<8> we_a[8];
	ACC_DTYPE<8> we_b[8];
	ACC_DTYPE<32> prod[16][4];

#pragma HLS array_partition variable=lhs_read complete dim=0
#pragma HLS array_partition variable=rhs_read complete dim=0
#pragma HLS array_partition variable=in_a complete dim=0
#pragma HLS array_partition variable=in_b complete dim=0
#pragma HLS array_partition variable=we_a complete dim=0
#pragma HLS array_partition variable=we_b complete dim=0
#pragma HLS array_partition variable=prod complete dim=0

	for(int i=0;i<4;i++){
#pragma HLS unroll
		for(int j=0;j<16;j++){
#pragma HLS unroll
			out[j][i] = 0;
		}
	}

	DWAIT(4);
	for (int rin=0;rin<d; rin++){
#pragma HLS pipeline II=1
		lhs_read[0] = l1[rin+l_pointer];
		lhs_read[1] = l2[rin+l_pointer];
		lhs_read[2] = l3[rin+l_pointer];
		lhs_read[3] = l4[rin+l_pointer];
		rhs_read[0] = r1[rin];
		rhs_read[1] = r2[rin];
		rhs_read[2] = r3[rin];
		rhs_read[3] = r4[rin];
		for(int i=0;i<4;i++){
#pragma HLS unroll
			in_a[i+0] = lhs_read[i].range(7,0);
			in_a[i+4] = lhs_read[i].range(15,8);
			in_b[i+0] = lhs_read[i].range(23,16);
			in_b[i+4] = lhs_read[i].range(31,24);

			we_a[i+0] = rhs_read[i].range(7,0);
			we_a[i+4] = rhs_read[i].range(15,8);
			we_b[i+0] = rhs_read[i].range(23,16);
			we_b[i+4] = rhs_read[i].range(31,24);
		}
		for(int i=0;i<4;i++){
#pragma HLS unroll
			prod[i*4+0][0] = in_a[0*4+i] * we_a[0*4+0];
			prod[i*4+1][0] = in_a[0*4+i] * we_a[0*4+1];
			prod[i*4+2][0] = in_a[0*4+i] * we_a[0*4+2];
			prod[i*4+3][0] = in_a[0*4+i] * we_a[0*4+3];
			prod[i*4+0][1] = in_a[1*4+i] * we_a[1*4+0];
			prod[i*4+1][1] = in_a[1*4+i] * we_a[1*4+1];
			prod[i*4+2][1] = mul_s8(in_a[1*4+i],we_a[1*4+2]);
			prod[i*4+3][1] = mul_s8(in_a[1*4+i],we_a[1*4+3]);
			prod[i*4+0][2] = mul_s8(in_b[0*4+i],we_b[0*4+0]);
			prod[i*4+1][2] = mul_s8(in_b[0*4+i],we_b[0*4+1]);
			prod[i*4+2][2] = mul_s8(in_b[0*4+i],we_b[0*4+2]);
			prod[i*4+3][2] = mul_s8(in_b[0*4+i],we_b[0*4+3]);
			prod[i*4+0][3] = mul_s8(in_b[1*4+i],we_b[1*4+0]);
			prod[i*4+1][3] = mul_s8(in_b[1*4+i],we_b[1*4+1]);
			prod[i*4+2][3] = mul_s8(in_b[1*4+i],we_b[1*4+2]);
			prod[i*4+3][3] = mul_s8(in_b[1*4+i],we_b[1*4+3]);
		}
		for(int i=0;i<16;i++){
#pragma HLS unroll
			out[i][0] += prod[i][0];
			out[i][1] += prod[i][1];
			out[i][2] += prod[i][2];
			out[i][3] += prod[i][3];
		}
#ifndef __SYNTHESIS__
		gmacs->array[wID]+=64;
		DWAIT();
#endif
	}
	for(int i=0;i<16;i++){
#pragma HLS unroll
		out[i][0] += out[i][1] + out[i][2] + out[i][3];
	}
	DWAIT(2);
}
