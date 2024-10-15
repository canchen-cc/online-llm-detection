#!/usr/bin/env bash
# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/raw_data
res_path=$exp_path/data
mkdir -p $exp_path $data_path $res_path

datasets="xsum olympic"
source_models="gemini_1.5_flash gemini_1.5_pro palm2"

# Black-box Setting
echo `date`, Evaluate models in the black-box setting:
scoring_models="gemma_2b"

# evaluate Fast-DetectGPT
for D in $datasets; do
  for M in $source_models; do
    M1=gemma_2b  # sampling model
    for M2 in $scoring_models; do
      echo `date`, Evaluating Fast-DetectGPT on ${D}.${M}.${M1}.${M2} ...
      python scripts/fast_detect_gpt.py --reference_model_name ${M1} --scoring_model_name ${M2} --dataset $D \
                          --dataset_file $data_path/${D}.${M} --output_file $res_path/${D}.${M}.${M2}
    done
  done
done

# evaluate DetectGPT and its improvement DetectLLM (NPR, LRR)
for D in $datasets; do
  for M in $source_models; do
    M1=t5-3b  # perturbation model
    for M2 in $scoring_models; do
      echo `date`, Evaluating DetectGPT on ${D}.${M}.${M1}.${M2} ...
      python scripts/detect_gpt.py --mask_filling_model_name ${M1} --scoring_model_name ${M2} --n_perturbations 100 --dataset $D \
                          --dataset_file $data_path/${D}.${M} --output_file $res_path/${D}.${M}.${M2}
      # we leverage DetectGPT to generate the perturbations
      echo `date`, Evaluating DetectLLM methods on ${D}.${M}.${M1}.${M2} ...
      python scripts/detect_llm.py --scoring_model_name ${M2} --dataset $D \
                          --dataset_file $data_path/${D}.${M}.${M1}.perturbation_100 --output_file $res_path/${D}.${M}.${M2}
    done
  done
done

# evaluate Logrank, Likelihood, Entropy
for D in $datasets; do
  for M in $source_models; do
    for M2 in $scoring_models; do
      echo `date`, Evaluating Logrank, Likelihood, Entropy on ${D}.${M}.${M2} ...
      python scripts/baselines.py --output_file $res_path/${D}.${M}.${M2}\
        --dataset_file $data_path/${D}.${M} --scoring_model_name ${M2} --device 'cuda'\
        --cache_dir '../cache' --hf_token 'xxxxx'
    done
  done
done



# evaluate DNA-GPT
for D in $datasets; do
  for M in $source_models; do
    for M2 in $scoring_models; do
      echo `date`, Evaluating Fast-DetectGPT on ${D}.${M}.${M2} ...
      python scripts/dna_gpt.py --output_file $res_path/${D}.${M}.${M2}\
        --dataset_file $data_path/${D}.${M} --base_model_name ${M2} --device 'cuda'\
        --cache_dir '../cache' --hf_token 'xxxxx'
    done
  done
done


for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating roberta-base on ${D}.${M} ...
    python scripts/supervised.py --output_file $res_path/${D}.${M}\
            --dataset_file $data_path/${D}.${M} --model_name 'roberta-base-openai-detector' --device 'cuda' --cache_dir '../cache'
    echo `date`, Evaluating roberta-large on ${D}.${M} ...
    python scripts/supervised.py --output_file $res_path/${D}.${M}\
        --dataset_file $data_path/${D}.${M} --model_name 'roberta-large-openai-detector' --device 'cuda' --cache_dir '../cache'
  done
done
