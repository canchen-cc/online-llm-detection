# GPT-Detection-via-Online-Optimization

Our test is conducted in the black-box setting, which means the scoring model is different from the source model. 3 souce models are applied to generate equal number of fake news based on the real 2024 Olympic News, including Gemini-1.5-Flash, Gemini-1.5-Pro and PaLM 2. There are 10 score functions: Fast-DetectGPT (sampling_discrepancy), DetectGPT (perturbation_100), LRR, NPR, Likelihood, Logrank, Entropy, DNA-GPT, RoBERTa-base, RoBERTa-large. For the fist 8 score functions, the scoring models Neo2.7 and Gemma-2B are used. Human-written texts $x_t$ are sampled from XSum dataset.

Firstly, we produce machine-generated-texts by source models. Then, we score human-written-texts and machine-generated  and real news by the above score functions. Next, we input 
