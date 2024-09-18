%%bash
#fastdetect
# Setup the environment 
echo "$(date), Setting up environment ..."
mkdir -p results  # Create the results directory if it does not exist

# Input scores evaluated by Fast-DetectGPT
python scripts/detect_case1.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.gemma_2b.sampling_discrepancy.json" --type1 "real" \  # sequence of human texts x_t
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.sampling_discrepancy.json" --type2 "samples" \ # sequence of LLM texts y_t
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.sampling_discrepancy.json" --type3 "real" \ # sequence of human texts y_t
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"


# Input scores evaluated by DetectGPT
python scripts/bet_random_real_fpr_composite_null_estimate5.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.gemma_2b.perturbation_100.json" --type1 "real" \
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.perturbation_100.json" --type2 "samples" \
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.perturbation_100.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"

# Input scores evaluated by NPR
python scripts/bet_random_real_fpr_composite_null_estimate5.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.gemma_2b.npr.json" --type1 "real" \
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.npr.json" --type2 "samples" \
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.npr.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"

# Input scores evaluated by LRR
python scripts/bet_random_real_fpr_composite_null_estimate5.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.gemma_2b.lrr.json" --type1 "real" \
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.lrr.json" --type2 "samples" \
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.lrr.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"

# Input scores evaluated by Logrank
python scripts/bet_random_real_fpr_composite_null_estimate5.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.gemma_2b.logrank.json" --type1 "real" \
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.logrank.json" --type2 "samples" \
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.logrank.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"

# Input scores evaluated by Likelihood
python scripts/bet_random_real_fpr_composite_null_estimate5.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.gemma_2b.likelihood.json" --type1 "real" \
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.likelihood.json" --type2 "samples" \
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.likelihood.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"

# Input scores evaluated by Entropy
python scripts/bet_random_real_fpr_composite_null_estimate5.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.gemma_2b.entropy.json" --type1 "real" \
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.entropy.json" --type2 "samples" \
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.entropy.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"

# Input scores evaluated by DNA-GPT
python scripts/bet_random_real_fpr_composite_null_estimate5.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.gemma_2b.dna_gpt.json" --type1 "real" \
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.dna_gpt.json" --type2 "samples" \
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.gemma_2b.dna_gpt.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"


# Input scores evaluated by RoBERTa-base 
python scripts/bet_random_real_fpr_composite_null_estimate5.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.roberta-base-openai-detector.json" --type1 "real" \
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.roberta-base-openai-detector.json" --type2 "samples" \
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.roberta-base-openai-detector.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"

# Input scores evaluated by RoBERTa-large 
python scripts/bet_random_real_fpr_composite_null_estimate5.py --file1 "/.../data/olympic_test/xsum.flash.gemma/xsum.gemini_1.5_flash.roberta-large-openai-detector.json" --type1 "real" \
                             --file2 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.roberta-large-openai-detector.json" --type2 "samples" \
                             --file3 "/.../data/olympic_test/olympic.flash.gemma/olympic.gemini_1.5_flash.roberta-large-openai-detector.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "/.../results/olympic_test_new/case6_estimate5/flash.gemma.500.json"














