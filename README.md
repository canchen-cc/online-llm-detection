# ONLINE DETECTING LLM-GENERATED TEXTS
**This code is for paper "ONLINE DETECTING LLM-GENERATED TEXTS VIA SEQUENTIAL HYPOTHESIS TESTING BY BETTING "**, where we borrow or extend some code from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) and [auditing-fairness](https://github.com/bchugg/auditing-fairness).

## Brief Info
Our method detects the source of texts observed in a streaming fashion. Tests are implemented in a black-box setting, meaning the model used for scoring texts is different from the source model used to generate them.

## Score Functions & Scoring Models
Our method builds upon and complements existing offline detection techniques. Specifically, We utilize 10 score functions of existing detectors to compute the scores of texts:
* Fast-DetectGPT (sampling_discrepancy/sampling_discrepancy_analytic)
* DetectGPT (perturbation_100)
* LRR 
* NPR 
* Likelihood
* Logrank
* Entropy
* DNA-GPT
* RoBERTa-base
* RoBERTa-large

The scoring models GPT-Neo-2.7B and Gemma-2B are applied to eveluate metrics involved in the first 8 score functions listed above.

## Environment
* Python3.10
* PyTorch2.4.0
* Setup the environment:
  ```bash setup.sh```
  
(Notes: our experiments are run on 1 GPU of Tesla A100 with 40G memory.)

## Workspace
Our experiments are organized into the following directories:
- **`./exp_main`**: Contains experiments for detecting Paris 2024 Olympic news or fake Olympic news generated by LLMs, including Gemini-1.5-Flash, Gemini-1.5-Pro, and PaLM 2. The scoring models used are GPT-Neo-2.7B and Gemma-2B.

- **`./exp_gpt3to4`**: Contains experiments for detecting human-written texts versus LLM-generated texts of various domains, based on the existing dataset from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt). Source models include GPT-3, ChatGPT, and GPT-4, with GPT-Neo-2.7B as the scoring model.

Both `./exp_main` and `./exp_gpt3to4` include the following common subdirectories:
- **`/raw_data`**: Stores human-written texts alongside an equal number of texts generated by the source models based on the first 30 tokens of the human-written texts.
- **`/data`**: Includes scores of both human-written and LLM-generated texts.
- **`/results`**: Contains raw results from the detection of human-written versus LLM-generated texts using specific score functions under each configuration, results of statistical metrics of 10 score functions under the same configurations used for plotting, as well as the final plotting results.
  
## Scenarios
Two scenarios are considered:
* **Scenario 1 (oracle):** Assumes prior knowledge of parameters \(d_*\) and ε.
* **Scenario 2:** Specifies the value of ε based on 20 human-written texts. The value of \(d_*\) (or \(d_t\)) is estimated using samples collected in the first 10 time steps, after which hypothesis testing starts.

## Process
* **Firstly**, we use source models to generate an equal number of texts based on each human-written text. (run_to_generate_main.sh)
* **Next**, we use score functions to score all the human-written and LLM-generated texts. (run_to_score.sh)
* **Then**, we use our methods to detect the source of a sequence of texts under both H0 and H1. (run_to_detect.sh)
* **For each significance level**, we calculate the false positive rate (FPR) if the source is human; otherwise, we report the rejection time, i.e., the time step at which the source is declared to be an LLM. (plot_results.ipynb)


### Citation
If you find this work useful, you can cite it with this BibTex entry:

    @misc{chen2024onlinedetectingllmgeneratedtexts,
          title={Online Detecting LLM-Generated Texts via Sequential Hypothesis Testing by Betting}, 
          author={Can Chen and Jun-Kun Wang},
          year={2024},
          eprint={2410.22318},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2410.22318}, 
    }

   
