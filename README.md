# InferDPT

Code for  "InferDPT: Privacy-preserving Inference for Black-box Large Language Models"

Note that this repo is anonymous and only intended for **review** purpose only. 

## Introduction

We propose InferDPT, the first practical framework for privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. We also propose RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency list" for TEXT perturbation within the prompt.

<p align="center">
<img src="img.png" alt="" width="800" title="The overview of InferDPT. The Prefix text is the perturbed document 𝐷𝑜𝑐𝑝 via RANTEXT. We use the same color to mark the perturbed parts in the raw document and the perturbed document. We also use the same color to highlight identical text appearing in the perturbed generation result, the extraction generation result, and the non-private generation result.
"/>
</p>

## Setup Environment

### Install required packages

#### step1  Install RANTEXT

```shell
git clone https://github.com/mengtong0110/InferDPT
pip install -r requirements.txt
```

#### step2  [Install FastChat](https://github.com/lm-sys/FastChat)

```
pip install "fschat[model_worker,webui]"
```

#### step3  [Install GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa)

```
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git repositories/GPTQ-for-LLaMa
cd repositories/GPTQ-for-LLaMa
git switch fastest-inference-4bit
python3 setup_cuda.py install
pip3 install texttable
```



## Privacy-preserving Inference with InferDPT

#### step1. Perturbation Module

Run the following code to get the Perturbed Generation:

```
python main.py  #You need to modify the variable to your input data (Prefix Text) and get the Perturbed Generation.
```

#### step2. Extraction Module

Deploy a model locally and use the following prompt to complete the text generation task:

```
Your task is to extend the “Prefix Text”. Use the “Perturbed Generation” as your primary writing material for your extension. Extract
coherent and consistent text from the “Perturbed Generation” and
integrate them into your continuation. Ensure a seamless alignment
with the context established by the “Prefix Text”. Provide only your
“Extended Text”
——“Prefix Text”:
——“Perturbed Generation”:
——“Extended Text”:
```

For information about model deployment, please refer to [FastChat](https://github.com/lm-sys/FastChat) and [GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa).