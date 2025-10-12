# Replication Package: Does Model Size Matter? A Comparison of Small and Large Language Models for Requirements Classification

This repository contains the replication package for the research paper:

**"Does Model Size Matter? A Comparison of Small and Large Language Models for Requirements Classification"**

*Mohammad Amin Zadenoori¹, Vincenzo De Martino²³, Jacek Dąbrowski⁴, Xavier Franch³, and Alessio Ferrari⁵*

¹ University of Padova, Italy  
² Software Engineering (SeSa) Lab, University of Salerno, Italy  
³ Universitat Politècnica de Catalunya, Spain  
⁴ Lero, the Research Ireland Centre for Software, University of Limerick, Ireland  
⁵ University College Dublin (UCD), Ireland  

## Abstract

**[Context and motivation]** Large language models (LLMs) show promising results in natural language processing (NLP) tasks for requirements engineering (RE). However, their use is limited by high computational cost, data-sharing risks, and dependence on external services. Small language models (SLMs) offer a lightweight, locally deployable alternative. 

**[Question/problem]** However, it remains unclear how well open-source SLMs perform compared to LLMs in RE tasks. 

**[Principal idea/results]** This preliminary study compares eight models, including three LLMs and five SLMs, on the task of requirements classification, using three datasets. We evaluated performance using precision, recall, and F1 metrics to assess their ability to classify functional vs non-functional, functional vs quality, and security vs non-security requirements. Results indicate that LLMs generalize slightly better. The best-performing LLMs achieved approximately 2% higher F1 scores compared to the top-performing SLMs across the evaluated classification tasks, while SLMs have lower computational demands and do not share sensitive data. 

**[Contributions]** Our contribution is twofold: (i) a systematic comparison of five SLMs and three LLMs, and (ii) evidence that SLMs can reliably support RE classification tasks on local infrastructure.

## Project Structure

```
.
├── input/                          # Input datasets used in the study
│   ├── CPN.csv                    # Security classification dataset
│   └── promise-reclass.csv        # PROMISE reclassification dataset
├── output/                        # Generated results and metrics
│   ├── all_summary_metrics_nfr.csv
│   ├── all_summary_metrics_promise_refined.csv
│   └── all_summary_metrics_security.csv
├── prompts and descriptions/      # Prompt templates and class descriptions
│   └── Prompt and descriptions.md
├── ReqSec.py                      # Security requirements classification
├── NFR.py                         # Functional vs Non-Functional classification
├── Promise-reclass.py             # F, Q, onlyF, onlyQ classification
└── README.md                      # This file
```

## Research Components

### Models Evaluated

#### Small Language Models (SLMs)
- Qwen/Qwen2-7B-Instruct
- tiiuae/Falcon3-7B-Instruct
- ibm-granite/granite-3.2-8b-instruct
- mistralai/Ministral-8B-Instruct-2410
- meta-llama/Meta-Llama-3-8B-Instruct

#### Large Language Models (LLMs)
- Grok-4
- GPT-5o (using OpenAI API)
- Claude-4 (using Anthropic API)

### Classification Tasks

#### 1. Security Requirements Classification (`ReqSec.py`)
- **Task**: Binary classification of requirements as Security vs Non-Security
- **Dataset**: CPN.csv
- **Key Finding**: LLMs achieved ~2% higher F1 scores compared to top SLMs

#### 2. Functional vs Non-Functional Requirements (`NFR.py`)
- **Task**: Binary classification of requirements as Functional vs Non-Functional
- **Dataset**: promise-reclass.csv
- **Key Finding**: SLMs demonstrated competitive performance with local deployment advantages

#### 3. Fine-grained Requirements Classification (`Promise-reclass.py`)
- **Task**: Multi-class classification into Functional (F), Quality (Q), Only Functional (onlyF), and Only Quality (onlyQ)
- **Dataset**: promise-reclass.csv
- **Key Finding**: Both SLMs and LLMs effectively handle complex multi-class classification

## Experimental Setup

### Prompting Strategies
Each classification task was evaluated using four prompting methods:
1. **CoT with Few-shot**: Combined approach with examples and reasoning

### Evaluation Metrics
- Precision (P)
- Recall (R) 
- F1-score

### Key Experimental Features
- Dynamic batching based on GPU memory
- Majority voting with consensus mechanism
- Variance checking for prediction stability
- Comprehensive error handling

## Installation and Setup

### Prerequisites
```bash
pip install torch transformers datasets scikit-learn pandas numpy requests openai
```

### Environment Variables
Set the following API keys for LLM integration:

```bash
# For Grok API
export GROK_API_KEY="your_grok_api_key"

# For Claude API
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# For OpenAI API (GPT-5o)
export OPENAI_API_KEY="your_openai_api_key"

# Hugging Face token (for local models)
export HF_TOKEN="your_huggingface_token"
```

## Usage

### Reproducing Security Classification Experiments
```bash
python ReqSec.py
```

### Reproducing Functional vs Non-Functional Classification
```bash
python NFR.py
```

### Reproducing Fine-grained Classification
```bash
python Promise-reclass.py
```

## Key Findings Reproduction

The experiments demonstrate that:

1. **LLMs show slightly better generalization** with approximately 2% higher F1 scores across tasks
2. **SLMs provide practical alternatives** with local deployment and data privacy advantages
3. **Both model types effectively handle** complex requirements classification tasks
4. **The performance gap is minimal** for many practical applications

## Output Interpretation

Each script generates comprehensive CSV files containing:

- Overall metrics (Precision, Recall, F1)
- Class-specific performance metrics
- Timing data for computational efficiency analysis
- Comparative performance between SLMs and LLMs

## Dataset Information

### CPN.csv
- Security requirements classification dataset
- Binary labels: Security (1) vs Non-Security (0)

### promise-reclass.csv  
- PROMISE requirements repository with reclassified labels
- Supports multiple classification schemes including functional vs non-functional and quality attributes

## 📚 Dataset Attribution

This project uses the following publicly available datasets:

- **PROMISE Dataset**  
  Originally provided for the RE'17 Data Challenge by Jane Cleland-Huang and collaborators.  
  - Cleland-Huang, J., Mazrouee, S., Huang, L., & Port, D. (2007). *nfr* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.268542](https://doi.org/10.5281/zenodo.268542)  
  - RE'17 Data Challenge: [http://ctp.di.fct.unl.pt/RE2017/pages/submission/data_papers/](http://ctp.di.fct.unl.pt/RE2017/pages/submission/data_papers/)  
  - See also: Sayyad Shirabad, J. & Menzies, T.J. (2005). *The PROMISE Repository of Software Engineering Databases*. University of Ottawa. [http://promise.site.uottawa.ca/SERepository](http://promise.site.uottawa.ca/SERepository)

- **Relabeled PROMISE Dataset**  
  Provided by Dalpiaz et al. for explainable requirements classification.  
  - Dalpiaz, F., Dell’Anna, D., Aydemir, F. B., & Çevikol, S. (2019). *explainable-re/re-2019-materials*. Zenodo. [https://doi.org/10.5281/zenodo.3309669](https://doi.org/10.5281/zenodo.3309669)

- **SecReq Dataset**  
  Used for security requirements classification.  
  - Knauss, E., Houmb, S. H., Islam, S., Jürjens, J., & Schneider, K. (2021). *SecReq* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.4530183](https://doi.org/10.5281/zenodo.4530183)


## Citation

If you use this replication package in your research, please cite our paper:

```bibtex
@article{zadenoori2025model,
  title={Does Model Size Matter? A Comparison of Small and Large Language Models for Requirements Classification},
  author={Zadenoori, Mohammad Amin and De Martino, Vincenzo and Dąbrowski, Jacek and Franch, Xavier and Ferrari, Alessio},
  journal={Submitted for publication},
  year={2025}
}
```

## Contact

For questions about this replication package, please contact:
- Mohammad Amin Zadenoori: amin.zadenoori@unipd.it

## License

[Add appropriate license information based on your institution's policies]
