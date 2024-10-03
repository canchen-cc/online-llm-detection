# ONLINE DETECTING LLM-GENERATED TEXTS
**This code is for ICLR 2025 paper "ONLINE DETECTING LLM-GENERATED TEXTS VIA SEQUENTIAL HYPOTHESIS TESTING BY BETTING "**, where we borrow or extend some code from [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) and [auditing-fairness](https://github.com/ bchugg/auditing-fairness).

[Paper](url) 
| [LocalDemo](#local-demo)
| [OnlineDemo](http://region-9.autodl.pro:21504/)
| [OpenReview](https://openreview.net/forum?id=Bpcgcr8E8Z)

## Brief Info
Our test is conducted in the black-box setting, which means the scoring model is different from the source model. 3 souce models are applied to generate equal number of fake news based on the real 2024 Olympic News, including Gemini-1.5-Flash, Gemini-1.5-Pro and PaLM 2. There are 10 score functions: Fast-DetectGPT (sampling_discrepancy), DetectGPT (perturbation_100), LRR, NPR, Likelihood, Logrank, Entropy, DNA-GPT, RoBERTa-base, RoBERTa-large. For the fist 8 score functions, the scoring models Neo2.7 and Gemma-2B are used. Human-written texts $x_t$ are sampled from XSum dataset.

## Environment
* Python3.10
* PyTorch2.4.0
* Setup the environment:
  ```bash setup.sh```
  
(Notes: our experiments are run on 1 GPU of Tesla A100 with 80G memory.)

## Workspace
Folders created for our experiments include:
* ./exp_main -> experiments for detecting Paris 2024 Olympic News or fake news about it. Fake news are generated by Gemini-1.5-Flash, Gemini-1.5-Pro and PaLM 2, scoring models used to calculate the value of eight score functions are GPT-Neo-2.7B and Gemma-2B.
* ./exp_gpt3to4 -> experiments for detecting human-written or LLM-generated texts based on the exsiting dataset of [Fast-DetectGPT] (https://github.com/baoguangsheng/fast-detect-gpt). Source models of LLM-generated texts are GPT-3, ChatGPT and GPT-4, the scoring model used to calculate the value of eight score functions is GPT-Neo-2.7B.
  
Firstly, we produce machine-generated-texts by source models. Then, we score human-written-texts and machine-generated  and real news by the above score functions. Next, we input 
