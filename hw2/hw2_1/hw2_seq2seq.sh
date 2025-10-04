#!/bin/bash
# HW2 Seq2Seq Shell Script
# Usage: ./hw2_seq2seq.sh <testing_data_directory> <output_filename>
# Example: ./hw2_seq2seq.sh testing_data testset_output.txt

#check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments"
    echo "Usage: ./hw2_seq2seq.sh <testing_data_directory> <output_filename>"
    echo "Example: ./hw2_seq2seq.sh testing_data testset_output.txt"
    exit 1
fi

#assign arguments
TEST_DIR=$1
OUTPUT_FILE=$2

#model path 
MODEL_PATH="rc_seq2seq_model"

echo "=========================================="
echo "HW2 - Seq2Seq Model for Video Captioning"
echo "=========================================="
echo "Testing directory: $TEST_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Model path: $MODEL_PATH"
echo "=========================================="

#check if model weights exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found at $MODEL_PATH"
    exit 1
fi

#run inference using model_seq2seq.py
/usr/bin/python3 model_seq2seq.py \
    "$TEST_DIR" \
    "$OUTPUT_FILE" \
    --model_path "$MODEL_PATH"

#check if script succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: Saved to: $OUTPUT_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Inference failed"
    echo "=========================================="
    exit 1
fi

