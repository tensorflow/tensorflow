// This example was extracted from:
// https://stackoverflow.com/a/46357313/2971299
// This thread also includes other mothodologies for more elaborated
// testing enviroments.

#include <systemc/systemc.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

struct test_driver;

test_driver *test_driver_p = nullptr;

void register_test_driver(test_driver *td) {
    test_driver_p = td;
}

test_driver* get_test_driver() {
    assert(test_driver_p);
    return test_driver_p;
}


SC_MODULE(dut_accum) {
    sc_in_clk   clk{"clk"};
    sc_in<bool> reset{"reset"};

    sc_in<bool> en{"en"};
    sc_in<int>  din{"din"};
    sc_out<int> dout{"dout"};

    SC_CTOR(dut_accum) {
        SC_METHOD(accum_method);
        sensitive << clk.pos();
    };

    void accum_method() {
        if (reset)
            dout = 0;
        else if (en)
            dout = dout + din;
    }
};

SC_MODULE(test_driver) {

    sc_signal<bool> reset{"reset",1};
    sc_signal<bool> en{"en",0};
    sc_signal<int> din{"din",0};
    sc_signal<int> dout{"dout"};

    SC_CTOR(test_driver) {
        dut_inst.clk(clk);
        dut_inst.reset(reset);
        dut_inst.en(en);
        dut_inst.din(din);
        dut_inst.dout(dout);
        SC_THREAD(test_thread);
        sensitive << clk.posedge_event();
        register_test_driver(this);
    }

private:
    void test_thread() {
        if (RUN_ALL_TESTS())
            SC_REPORT_ERROR("Gtest", "Some test FAILED");
        sc_stop();
    }

    dut_accum dut_inst{"dut_inst"};
    sc_clock clk{"clk", 10, SC_NS};
};



namespace {
    // The fixture for testing dut_accum
    class accum_test: public ::testing::Test {
    protected:

        test_driver & td;

        accum_test(): td(*get_test_driver()){
            reset_dut();
        }

        virtual ~accum_test() {}

        void reset_dut(){
            td.reset = 1;
            wait();
            td.reset = 0;
        }
    };

    TEST_F(accum_test, test0) {
        td.din = 10;
        td.en = 1;
        wait();
        wait();
        EXPECT_EQ(td.dout.read(), 10);
    }

    // TEST_F(accum_test, test1_no_en) {
    //     td.din = 10;
    //     td.en = 0;
    //     wait();
    //     wait();
    //     EXPECT_EQ(td.dout.read(), 10); // this test will fail, since en is 0
    // }

    TEST_F(accum_test, test2_reset_asserted) {
        td.din = 10;
        td.en = 1;
        td.reset = 1;
        wait();
        wait();
        EXPECT_EQ(td.dout.read(), 0);
    }
}

int sc_main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    test_driver td{"td"};
    sc_start();
    return 0;
}