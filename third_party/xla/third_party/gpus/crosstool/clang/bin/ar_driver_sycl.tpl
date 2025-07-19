#!/usr/bin/env bash

# Creates a temporary working directory to extract contents from .a archive files.
# Automatically detects whether each input is a regular object file (.o) or an archive (.a).
# If the input is a regular object file (.o), it is included directly in the final archive.
# If the input is an archive (.a), it is first unpacked, and its extracted object files are then added to the final archive. 

# Create a temporary directory
tmp_dir="$(mktemp -d -t tmp.XXXXXXXXXX)"
input_object_file=""
ar_flag=""
output_file=""

if [[ $# -eq 1 ]]; then
    arg="$1"
    shift
    if [[ $arg == "@"* ]]; then
        file_name=${arg#*@}

        {
            read -r ar_flag
            read -r output_file
            while IFS= read -r input_file; do
                if file "$input_file" | grep -q "current ar archive"; then
                    ar x "$input_file" --output="$tmp_dir"
                else
                    input_object_file="$input_object_file $input_file"
                fi
            done
        } < "$file_name"
    else
        echo "invalid argument"
        exit 1
    fi
else
    ar_flag="$1"
    shift
    output_file="$1"
    shift

    for input_file in "$@"; do
        if file "$input_file" | grep -q "current ar archive"; then
            ar x "$input_file" --output="$tmp_dir"
        else
            input_object_file="$input_object_file $input_file"
        fi
    done
fi

if [[ $input_object_file != "" ]]; then
    ar "$ar_flag" "$output_file" $input_object_file
else
    ar "$ar_flag" "$output_file" "$tmp_dir"/*
fi

# Remove the temporary directory
rm -rf "$tmp_dir"
