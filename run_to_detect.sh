%%bash
# Here, we give an example to show how to run scripts to get our detection results.
# Setup the environment 
echo "$(date), Setting up environment ..."
mkdir -p results  # Create the results directory if it does not exist

# Input scores evaluated by Fast-DetectGPT
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type1 "real" \  # prepared human-written texts x_t
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type2 "samples" \ # the sequence of LLM-generated texts y_t (the alternative hypothesis)
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.gemma_2b.sampling_discrepancy.json" --type3 "real" \ # the sequence of human-written texts y_t (the null hypothesis)
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"


# Input scores evaluated by DetectGPT
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.perturbation_100.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.perturbation_100.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.perturbation_100.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by NPR
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.npr.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.npr.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.npr.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by LRR
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.lrr.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.lrr.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.lrr.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by Logrank
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.logrank.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.logrank.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.logrank.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by Likelihood
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.likelihood.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.likelihood.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.likelihood.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by Entropy
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.entropy.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.entropy.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.entropy.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by DNA-GPT
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.dna_gpt.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.dna_gpt.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.dna_gpt.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"


# Input scores evaluated by RoBERTa-base 
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.roberta-base-openai-detector.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.roberta-base-openai-detector.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.roberta-base-openai-detector.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by RoBERTa-large 
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.roberta-large-openai-detector.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.roberta-large-openai-detector.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.roberta-large-openai-detector.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "././results/raw_results/flash.gemma.scenario1.json"














