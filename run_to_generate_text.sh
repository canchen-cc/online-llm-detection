#!/bin/bash
# Here, we show how to run different scripts to generate fake Olympic news by 3 source models.

echo "Starting the script for gemini_1.5_flash_generate..."
# The source model is Gemini-1.5-Flash.
python scripts/gemini_1.5_flash_generate.py \
    --input_file "exp_main/raw_data/olympic2024.raw_data.json" \
    --output_file "exp_main/raw_data/olympic.gemini_1.5_flash.raw_data.json" \
    --project_id "xxx" \
    --model_name "gemini-1.5-flash-001"
echo "gemini_1.5_flash_generate finished."

echo "Starting the script for gemini_1.5_pro_generate..."
# The source model is Gemini-1.5-Pro.
python scripts/gemini_1.5_pro_generate.py \
    --input_file "exp_main/raw_data/olympic2024.raw_data.json" \
    --output_file "exp_main/raw_data/olympic.gemini_1.5_pro.raw_data.json" \
    --project_id "xxx" \
    --model_name "gemini-1.5-pro-001"
echo "gemini_1.5_pro_generate finished."

echo "Starting the script for palm_generate..."
# The source model is PaLM 2.
python scripts/jupyter/scripts/palm_generate.py \
    --input_file "exp_main/raw_data/olympic2024.raw_data.json" \
    --output_file "exp_main/raw_data/olympic.palm2.raw_data.json" \
    --project_id "xxx" \
    --model_name "text-bison@001"
echo "palm_generate finished."


