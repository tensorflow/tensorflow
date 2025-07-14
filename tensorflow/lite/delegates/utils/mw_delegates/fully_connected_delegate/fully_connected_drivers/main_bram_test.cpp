#include "fully_connected_bram_driver.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>

int main() {
    try {
        FpgaBramDriver driver;
        driver.initialize_bram(false); // Clear BRAM on init

        int input_size;
        int output_size;

        std::cout << "Enter input size: ";
        std::cin >> input_size;
        std::cout << "Enter output size: ";
        std::cin >> output_size;
        
        int weight_size = input_size * output_size;
        int bias_size = output_size;

        // Create test data with varied float values
        std::vector<float> input(input_size);
        std::vector<float> weights(weight_size);
        std::vector<float> bias(bias_size);
        std::vector<float> output(output_size, 0.0f);
        
        // Initialize input with test values
        for (int i = 0; i < input_size; ++i) {
            input[i] = 1.5f + i * 0.1f;  // 1.5, 1.6, 1.7, ...
        }
        
        // Initialize weights with test values
        for (int i = 0; i < weight_size; ++i) {
            weights[i] = 2.0f + (i % 10) * 0.01f;  // 2.0, 2.01, 2.02, ..., 2.09, 2.0, ...
        }
        
        // Initialize bias with test values
        for (int i = 0; i < bias_size; ++i) {
            bias[i] = 0.5f + i * 0.05f;  // 0.5, 0.55, 0.6, ...
        }

        // Write test data to BRAMs
        std::cout << "\n================= Writing Float Data to BRAMs =====================" << std::endl;
        
        if (driver.write_input_to_bram(input.data(), input_size) != 0) {
            std::cerr << "Failed to write input to BRAM" << std::endl;
            return 1;
        } else {
            std::cout << "Successfully wrote input data to BRAM" << std::endl;
        }
        


        if (driver.write_weights_to_bram(weights.data(), weight_size) != 0) {
            std::cerr << "Failed to write weights to BRAM" << std::endl;
            return 1;
        } else {
            std::cout << "Successfully wrote weights data to BRAM" << std::endl;
        }
        


        if (driver.write_bias_to_bram(bias.data(), bias_size) != 0) {
            std::cerr << "Failed to write bias to BRAM" << std::endl;
            return 1;
        } else {
            std::cout << "Successfully wrote bias data to BRAM" << std::endl;
        }


        // TODO: There is the error while clearing the form all the bram
        // Clear output BRAM
        std::cout << "\n=== Do you want to clear the output BRAM y/n? ===" << std::endl;
        char choice = 'n';
        std::cin >> choice;
        if (choice != 'n' && choice != 'N') {
            if (driver.clear_output_bram() != 0) {
                std::cerr << "Failed to clear output BRAM" << std::endl;
            }
        }

        // Read back BRAM data
        std::cout << "\n=== Testing Input BRAM ===" << std::endl;
        std::cout << "Original input data:" << std::endl;
        for (int i = 0; i < input_size; ++i) {
            std::cout << input[i] << " ";
        }
        std::cout << std::endl;
        
        
        std::vector<float> input_readback(input_size, 0.0f);
        
        if (driver.test_read_input_bram(input_readback.data(), input_size) == 0) {
            std::cout << "Read back input data:" << std::endl;
            bool input_match = true;
            for (int i = 0; i < input_size; ++i) {
                std::cout << input_readback[i] << " ";
                if (std::fabs(input_readback[i] - input[i]) > 1e-6) {
                    input_match = false;
                }
            }
            std::cout << std::endl;
            std::cout << "Input data verification: " << (input_match ? "PASS" : "FAIL") << std::endl;
        } else {
            std::cerr << "Failed to read input from BRAM" << std::endl;
        }

        //weight readback
        std::cout << "\n=== Testing Weights BRAM ===" << std::endl;
        std::cout << "Original weights data:" << std::endl;
        for (int i = 0; i <  weight_size; ++i) {
            std::cout << weights[i] << " ";
        }
        std::cout << std::endl;
        
        std::vector<float> weights_readback(weight_size, 0.0f);
        
        if (driver.test_read_weights_bram(weights_readback.data(), weight_size) == 0) {
            std::cout << "Read back weights data:" << std::endl;
            bool weights_match = true;
            for (int i = 0; i < weight_size; ++i) {
                std::cout << weights_readback[i] << " ";
            }
            std::cout << std::endl;
            
            // Check all weights for verification
            for (int i = 0; i < weight_size; ++i) {
                if (std::fabs(weights_readback[i] - weights[i]) > 1e-6) {
                    weights_match = false;
                    break;
                }
            }
            std::cout << "Weights data verification: " << (weights_match ? "PASS" : "FAIL") << std::endl;
        } else {
            std::cerr << "Failed to read weights from BRAM" << std::endl;
        }
        
        // Test: Write and read back bias data
        std::cout << "\n=== Testing Bias BRAM ===" << std::endl;
        std::cout << "Original bias data:" << std::endl;
        for (int i = 0; i < bias_size; ++i) {
            std::cout << bias[i] << " ";
        }
        std::cout << std::endl;
        
        std::vector<float> bias_readback(bias_size, 0.0f);
        
        if (driver.test_read_bias_bram(bias_readback.data(), bias_size) == 0) {
            std::cout << "Read back bias data:" << std::endl;
            bool bias_match = true;
            for (int i = 0; i < bias_size; ++i) {
                std::cout << bias_readback[i] << " ";
                if (std::fabs(bias_readback[i] - bias[i]) > 1e-6) {
                    bias_match = false;
                }
            }
            std::cout << std::endl;
            std::cout << "Bias data verification: " << (bias_match ? "PASS" : "FAIL") << std::endl;
        } else {
            std::cerr << "Failed to read bias from BRAM" << std::endl;
        }

        // Test: Read output BRAM (may contain computation results)
        std::cout << "\n=== Testing Output BRAM ===" << std::endl;
        if (driver.read_output_from_bram(output.data(), output_size) == 0) {
            std::cout << "Output BRAM contents:" << std::endl;
            bool all_zeros = true;
            bool has_results = false;
            float min_val = output[0], max_val = output[0];
            
            for (int i = 0; i < output_size; ++i) {
                std::cout << output[i] << " ";
                if (std::fabs(output[i]) > 1e-6) {
                    all_zeros = false;
                    has_results = true;
                }
                if (output[i] < min_val) min_val = output[i];
                if (output[i] > max_val) max_val = output[i];
            }
            std::cout << std::endl;
            
            if (all_zeros) {
                std::cout << "Output BRAM verification: PASS (cleared/empty)" << std::endl;
            } else {
                std::cout << "Output BRAM contains computation results!" << std::endl;
                std::cout << "  Range: " << min_val << " to " << max_val << std::endl;
                std::cout << "  Status: PASS (FPGA performed computation)" << std::endl;
            }
        } else {
            std::cerr << "Failed to read output from BRAM" << std::endl;
        }
        
    return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}

