void ACCNAME::Input_Handler(){
inS.write(0);
int r_max;
int l_max;
DATA last ={5000,1};
wait();
  while(1){
	inS.write(1);
	ACC_DTYPE<32> header_lite= din1.read().data.to_int();

	// cout << header_lite << endl;
	// wait(5);
	// out_check.write(1);
	// wait();

	if(header_lite!=STOPPER){

		bool rhs_take=header_lite.range(0,0);
		bool lhs_take=header_lite.range(1,1);
		depth=header_lite.range(31,18);

		inS.write(2);
		ACC_DTYPE<32> hcounts=din1.read().data.to_int();

		int rhs_count = hcounts.range(15, 0);
		int lhs_count = hcounts.range(31, 16);

		inS.write(3);
		ACC_DTYPE<32> lengths=din1.read().data.to_int();
		
		int rhs_length = lengths.range(15, 0);
		int lhs_length = lengths.range(31, 16);

		if(rhs_take)ra=din1.read().data.to_int();

		d_in1.write(1);
		read_inputs.write(1);
		rtake.write(rhs_take);
		ltake.write(lhs_take);
		rlen.write(rhs_length);
		llen.write(lhs_length);
		DWAIT();

		inS.write(4);
		while(d_in1.read())wait();
		read_inputs.write(0);
		inS.write(5);
		wait();

		inS.write(6);
		if(lhs_take){
			r_max=rhs_count;
			l_max=lhs_count;
		}
// #ifndef __SYNTHESIS__
// 		if(rhs_take){
// 			out_check.write(1);
// 		}
// #endif

	}else{
		while(schedule.read())wait();
		rhs_block_max.write(r_max);
		lhs_block_max.write(l_max);
		rmax.write(r_max);
		lmax.write(l_max);
		schedule.write(1);
		out_check.write(1);
		inS.write(7);
		wait();
		while(schedule.read())wait();

	}


  }
}
