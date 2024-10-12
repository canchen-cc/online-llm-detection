#!/bin/bash
# Here, we give an example to show how to run scripts to get the detection results of 10 score functions.

# Setup the environment
echo "$(date), Setting up environment ..."
mkdir -p results  # Create the results directory if it does not exist

# file1[type1] is the prepared dataset of human-written texts, our goal is to detect the source of file2[type2] (under the alternative H1) with short rejection time, 
# and detect the source of file3[type3] (under the null H0) with desired FPR (below the significance level).

# Input scores evaluated by Fast-DetectGPT
python scripts/detect_scenario1.py --file1 "exp_main/data/xsum.gemini_1.5_flash.gemma_2b.sampling_discrepancy.json" --type1 "real" \  # the sequence of text x_t (prepared human-written texts)
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.sampling_discrepancy.json" --type2 "samples" \ # the sequence of y_t (the source is gemini-1.5-flash)
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.sampling_discrepancy.json" --type3 "real" \ # the sequence of y_t (the source is human)
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"


# Input scores evaluated by DetectGPT
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.perturbation_100.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.perturbation_100.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.perturbation_100.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by NPR
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.npr.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.npr.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.npr.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by LRR
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.lrr.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.lrr.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.lrr.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by Logrank
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.logrank.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.logrank.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.logrank.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by Likelihood
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.likelihood.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.likelihood.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.likelihood.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by Entropy
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.entropy.json" --type1 "real" \
                             --file2 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.entropy.json" --type2 "samples" \
                             --file3 "./exp_main/data/olympic.gemini_1.5_flash.gemma_2b.entropy.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by DNA-GPT
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.gemma_2b.dna_gpt.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.dna_gpt.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.gemma_2b.dna_gpt.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"


# Input scores evaluated by RoBERTa-base 
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.roberta-base-openai-detector.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.roberta-base-openai-detector.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.roberta-base-openai-detector.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"

# Input scores evaluated by RoBERTa-large 
python scripts/detect_scenario1.py --file1 "./exp_main/data/xsum.gemini_1.5_flash.roberta-large-openai-detector.json" --type1 "real" \
                             --file2 "exp_main/data/olympic.gemini_1.5_flash.roberta-large-openai-detector.json" --type2 "samples" \
                             --file3 "exp_main/data/olympic.gemini_1.5_flash.roberta-large-openai-detector.json" --type3 "real" \
                             --iters 1000 --shift_time None --output_file "exp_main/results/raw_results/flash.gemma.scenario1.json"


echo "Script execution completed."











