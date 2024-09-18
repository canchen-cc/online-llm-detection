%%bash
#fastdetect
# Setup the environment 
echo "$(date), Setting up environment ..."
mkdir -p results  # Create the results directory if it does not exist

# flashcess the audits(from two random data sources)
python scripts/detect_case1.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.gemma_2b.sampling_discrepancy.json" --type1 "real" \  # sequence of human texts x_t
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.sampling_discrepancy.json" --type2 "samples" \ # sequence of LLM texts y_t
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.sampling_discrepancy.json" --type3 "real" \ # sequence of human texts y_t
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"
