#!/bin/bash
# Here, we give an example to show how to run scripts to get the detection results of comparisons between our method and baselines. The score function is fast-detectgpt.

# Setup the environment
echo "$(date), Setting up environment ..."
mkdir -p results  # Create the results directory if it does not exist

# Note: file1[type1] is the prepared dataset of human-written texts, our goal is to detect the source of file2[type2] (under the alternative H1) with short rejection time, and detect the source of file3[type3] (under the null H0) with desired FPR (below the significance level).

# Our method  
python detect_scenario1.py --file1 "exp_main/data/xsum.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.baseline_no_correction.json"


# The permutation test with no correction to the significance level (batch_size k=25)
python baseline_no_correction.py --file1 "exp_main/data/xsum.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type3 "real" \
                             --k 25 --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.baseline_no_correction.json"

# The permutation test with no correction to the significance level (batch_size k=25)
python baseline_no_correction.py --file1 "exp_main/data/xsum.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type3 "real" \
                             --k 50 --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.baseline_no_correction.json"

# The permutation test with no correction to the significance level (batch_size k=25)
python baseline_no_correction.py --file1 "exp_main/data/xsum.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type3 "real" \
                             --k 100 --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.baseline_no_correction.json"

# The permutation test with no correction to the significance level (batch_size k=25)
python baseline_no_correction.py --file1 "exp_main/data/xsum.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type3 "real" \
                             --k 250 --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.baseline_no_correction.json"

# The permutation test with no correction to the significance level (batch_size k=25)
python baseline_no_correction.py --file1 "exp_main/data/xsum.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type3 "real" \
                             --k 500 --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.baseline_no_correction.json"

# The permutation test with no correction to the significance level (batch_size k=25)
python baseline_no_correction.py --file1 "exp_main/data/xsum.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type3 "real" \
                             --k 1000 --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.baseline_no_correction.json"

# Replace 'no_correction' with 'with_correction', we can get the results of comparisons between our method and the permutation test with correction to the significance level.

echo "Script execution completed."
