#!/bin/bash
# This script calculates compound interest.
# Usage: ./new-feature.sh
# Enter the principal amount, rate of interest, and number of years.
echo "Enter the principal amount:"
read principal
echo "Enter the annual interest rate (as a percentage):"
read rate
echo "Enter the number of years:"
read years
amount=$(echo "scale=2; $principal * (1 + $rate / 100) ^ $years" | bc)
echo "The amount after $years years is: $amount"
