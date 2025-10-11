import os
import re
import pandas as pd
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import time
import csv
import requests
import json
from openai import OpenAI

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIClient:
    """Base class for API clients"""
    
    def __init__(self):
        self.name = "BaseAPI"
    
    def classify_text(self, prompt, max_tokens=1024):
        raise NotImplementedError("Subclasses must implement this method")
    
    def batch_classify(self, prompts, batch_size=1):
        """Classify multiple prompts in batches"""
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            for prompt in batch:
                try:
                    result = self.classify_text(prompt)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error in {self.name} classification: {str(e)}")
                    batch_results.append("0")  # Default to negative class on error
            results.extend(batch_results)
            # Rate limiting delay
            time.sleep(0.1)
        return results

class GrokAPIClient(APIClient):
    """Grok API client (assuming similar interface to OpenAI)"""
    
    def __init__(self, api_key=None):
        super().__init__()
        self.name = "Grok"
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        if not self.api_key:
            logger.warning("Grok API key not found. Please set GROK_API_KEY environment variable.")
        
        # Initialize client (adjust base_url as needed for Grok API)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"  # Adjust this based on Grok's actual API endpoint
        )
    
    def classify_text(self, prompt, max_tokens=1024):
        if not self.api_key:
            return "0"
            
        try:
            response = self.client.chat.completions.create(
                model="grok-beta",  # Adjust model name as needed
                messages=prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Grok API error: {str(e)}")
            return "0"

class ClaudeAPIClient(APIClient):
    """Claude API client using Anthropic's API"""
    
    def __init__(self, api_key=None):
        super().__init__()
        self.name = "Claude"
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("Claude API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        
        # Anthropic API setup
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-sonnet-20240229"  # Adjust model as needed
    
    def classify_text(self, prompt, max_tokens=1024):
        if not self.api_key:
            return "0"
            
        try:
            # Convert to Anthropic's message format
            system_content = prompt[0]['content'] if prompt[0]['role'] == 'system' else ""
            user_messages = [msg for msg in prompt if msg['role'] == 'user']
            user_content = user_messages[0]['content'] if user_messages else ""
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "system": system_content,
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['content'][0]['text']
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return "0"

class GPT5APIClient(APIClient):
    """GPT-5 API client (assuming similar interface to OpenAI)"""
    
    def __init__(self, api_key=None):
        super().__init__()
        self.name = "GPT-5"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
    
    def classify_text(self, prompt, max_tokens=1024):
        if not self.api_key:
            return "0"
            
        try:
            # Note: Replace "gpt-5" with actual model name when available
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 as placeholder for GPT-5
                messages=prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-5 API error: {str(e)}")
            return "0"

# Initialize API clients
def initialize_api_clients():
    """Initialize all API clients"""
    clients = {}
    
    try:
        clients['grok'] = GrokAPIClient()
    except Exception as e:
        logger.warning(f"Failed to initialize Grok client: {str(e)}")
    
    try:
        clients['claude'] = ClaudeAPIClient()
    except Exception as e:
        logger.warning(f"Failed to initialize Claude client: {str(e)}")
    
    try:
        clients['gpt5'] = GPT5APIClient()
    except Exception as e:
        logger.warning(f"Failed to initialize GPT-5 client: {str(e)}")
    
    return clients

def initialize_pipeline(model_name):
    """Initialize the Hugging Face pipeline for a given model with model-specific handling."""
    try:
        logger.info(f"Attempting to load model: {model_name}")
        
        if model_name == "mistralai/Ministral-8B-Instruct-2410":
            logger.info(f"Initializing Mistral model with custom LLM settings")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=""
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                token=""
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.info(f"Set pad_token_id to eos_token_id ({tokenizer.eos_token_id}) for {model_name}")
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                return_full_text=False,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            logger.info(f"Mistral pipeline initialized successfully")
        
        elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
            logger.info(f"Initializing LLaMA model with direct pipeline")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                token="",
                legacy=True
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = 128009
                logger.info(f"Set pad_token_id to eos_token_id (128009) for {model_name}")
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                return_full_text=False,
                do_sample=False,
                token="",
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id
            )
            logger.info(f"LLaMA pipeline initialized successfully")
        
        else:
            logger.info(f"Initializing model {model_name} with AutoModelForCausalLM")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=""
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                token=""
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.info(f"Set pad_token_id to eos_token_id ({tokenizer.eos_token_id}) for {model_name}")
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                return_full_text=False,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            logger.info(f"Pipeline for {model_name} initialized successfully")
        
        pipeline.tokenizer.padding_side = "left"
        terminators = [
            tok for tok in [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ] if tok is not None
        ] or [pipeline.tokenizer.eos_token_id] if pipeline.tokenizer.eos_token_id is not None else []
        
        if not terminators:
            logger.warning(f"No terminators defined for {model_name}")
            return None, None
        
        return pipeline, terminators
    except Exception as e:
        logger.error(f"Failed to initialize pipeline for {model_name}: {str(e)}")
        return None, None

def get_dynamic_batch_size(model_name, available_memory, base_batch_size=16):
    """Estimate optimal batch size based on available GPU memory."""
    model_memory_estimates = {
        "Qwen/Qwen2-7B-Instruct": 14e9,
        "tiiuae/Falcon3-7B-Instruct": 14e9,
        "ibm-granite/granite-3.2-8b-instruct": 16e9,
        "mistralai/Ministral-8B-Instruct-2410": 16e9,
        "meta-llama/Meta-Llama-3-8B-Instruct": 16e9
    }
    model_memory = model_memory_estimates.get(model_name, 16e9)
    memory_per_sample = model_memory / base_batch_size
    free_memory = available_memory if available_memory > 0 else torch.cuda.mem_get_info()[0]
    max_batch_size = min(base_batch_size, int(free_memory / memory_per_sample * 0.9))
    return max(1, max_batch_size)

def check_prediction_variance(predictions):
    """Check if predictions are stable to reduce voting runs."""
    if len(predictions) < 2:
        return False
    votes = [np.bincount([int(p) for p in pred]) for pred in predictions]
    variance = np.var([v.argmax() for v in votes if v.size > 0])
    return variance < 0.1

def generate_prompts(df, class_examples, class_explanations, is_quality=False):
    """Generate prompts using class examples and explanations."""
    explanation_str = "\n".join([f'{label}: {explanation}' for label, explanation in class_explanations.items()])
    example_str = "\n".join([f'"{text}" --> {label}' for text, label in class_examples.items()])
    
    zero_shot_prompts = generate_all_prompts(df['text'].tolist(), explanations=explanation_str, is_quality=is_quality)
    few_shot_prompts = generate_all_prompts(df['text'].tolist(), examples=example_str, explanations=explanation_str, is_quality=is_quality)
    cot_prompts = generate_all_prompts(df['text'].tolist(), explanations=explanation_str, cot=True, is_quality=is_quality)
    cot_with_examples_prompts = generate_all_prompts(df['text'].tolist(), examples=example_str, explanations=explanation_str, cot=True, is_quality=is_quality)
    
    return zero_shot_prompts, few_shot_prompts, cot_prompts, cot_with_examples_prompts

def generate_messages(text, examples=None, explanations=None, cot=False, is_quality=False):
    system_content = f"""
As an expert system for classifying software requirements, your task is to carefully analyze each given requirement and categorize it into one of the following two categories:
"1": {'Quality' if is_quality else 'Functional'} (e.g., {'specifies performance, usability, reliability, or security attributes' if is_quality else 'defines specific system functions, behaviors, or operations'})
"0": {'Non-Quality' if is_quality else 'Non-Functional'} (e.g., {'describes functional behavior or implementation details' if is_quality else 'specifies performance, usability, reliability, or constraints'})
Output only the label in the format: "Label: [Your Class Label Here]". Do not provide any additional explanations.
""".strip()

    messages = [{"role": "system", "content": system_content}]
    if cot and examples:
        content = f"""
Let's analyze the classification step by step.
Step 1: Review the examples:
{examples}
Step 2: Understand the explanations:
{explanations}
Step 3: Apply this understanding to classify the following requirement:
Requirement: {text}
Step 4: Provide the final label in the format: "Label: [Your Class Label Here]".
"""
    elif examples:
        content = f"""
Examples:
{examples}
Explanations:
{explanations}
Requirement: {text}
Based on the examples and explanations above, classify the requirement and provide the final label in the format: "Label: [Your Class Label Here]".
"""
    else:
        content = f"""
Explanations:
{explanations}
Requirement: {text}
Using the explanations above, classify the requirement and provide the final label in the format: "Label: [Your Class Label Here]".
"""
    messages.append({"role": "user", "content": content.strip()})
    return messages

def generate_all_prompts(texts, examples=None, explanations=None, cot=False, is_quality=False):
    prompts = []
    for text in texts:
        messages = generate_messages(text, examples, explanations, cot, is_quality)
        prompts.append(messages)
    return prompts

def remove_assistant_prefix(data):
    extracted_data = []
    for item in data:
        text = item[0]['generated_text'] if isinstance(item, list) else item['generated_text']
        match = re.search(r'Label:\s*(\d)', text)
        extracted_data.append(match.group(1) if match else '0')
    return extracted_data

def mapping(outputs, labels):
    mapped = []
    for output in outputs:
        output_lower = output.strip().lower()
        found = False
        sorted_labels = sorted(labels.items(), key=lambda x: -len(x[0]))
        for label_key, label_value in sorted_labels:
            pattern = re.compile(r'\b' + re.escape(label_key.lower()) + r'\b')
            if pattern.search(output_lower):
                mapped.append(label_value)
                found = True
                break
        if not found:
            if '1' in output_lower:
                mapped.append(labels.get('1', '0'))
            elif '0' in output_lower:
                mapped.append(labels.get('0', '0'))
            else:
                mapped.append('0')
    return mapped

def compute_classification_metrics(y_true, y_pred, is_quality=False):
    target_names = ['Non-Quality', 'Quality'] if is_quality else ['Non-Functional', 'Functional']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    metrics = {
        target_names[1]: {
            'precision': report[target_names[1]]['precision'],
            'recall': report[target_names[1]]['recall'],
            'f1': report[target_names[1]]['f1-score'],
            'f2': fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
        },
        target_names[0]: {
            'precision': report[target_names[0]]['precision'],
            'recall': report[target_names[0]]['recall'],
            'f1': report[target_names[0]]['f1-score'],
            'f2': fbeta_score(y_true, y_pred, beta=2, pos_label=0, zero_division=0)
        },
        'avg': {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'f2': fbeta_score(y_true, y_pred, beta=2, average='weighted', zero_division=0)
        }
    }
    return metrics

def majority_vote(predictions):
    """Compute majority vote for a list of predictions."""
    voted = []
    for i in range(len(predictions[0])):
        votes = [pred[i] for pred in predictions]
        counts = np.bincount([int(v) for v in votes])
        if max(counts) >= len(predictions) // 2 + 1:
            voted.append(str(np.argmax(counts)))
        else:
            voted.append('0')  # Default to Non-Functional/Non-Quality
    return voted

def prepare_data(df):
    """Prepare datasets for Functional and Quality classifications with deduplication and overlap check."""
    df = df.drop_duplicates(subset=['text'], keep='first')
    logger.info(f"Dataset size after deduplication: {len(df)}")
    
    df_functional = df[['text', 'Functional']].rename(columns={'Functional': 'label'})
    df_quality = df[['text', 'Quality']].rename(columns={'Quality': 'label'})
    
    functional_examples = df_functional.drop_duplicates()
    quality_examples = df_quality.drop_duplicates()
    
    df_functional_train, df_functional_test = train_test_split(
        df_functional, test_size=0.7, random_state=42, stratify=df_functional['label']
    )
    df_quality_train, df_quality_test = train_test_split(
        df_quality, test_size=0.7, random_state=42, stratify=df_quality['label']
    )
    
    # Check for train-test overlap
    overlap_f = set(df_functional_train['text']).intersection(set(df_functional_test['text']))
    overlap_q = set(df_quality_train['text']).intersection(set(df_quality_test['text']))
    logger.info(f"Functional train-test overlap: {len(overlap_f)} texts")
    logger.info(f"Quality train-test overlap: {len(overlap_q)} texts")
    
    return (
        functional_examples, quality_examples,
        df_functional_train, df_functional_test,
        df_quality_train, df_quality_test
    )

def generate_summary_table_custom(all_metrics, num_runs):
    """Generate a simplified summary table with Average P, R, F1, F2 and F1, F2 for F, Q, onlyF, onlyQ."""
    methods = ['ZeroShot', 'FewShot', 'COT', 'COT_with_examples']
    classifications = [
        'Functional', 'Non-Functional', 
        'Quality', 'Non-Quality',
        'Functional_NonQuality', 'Non-Functional_NonQuality',
        'Quality_NonFunctional', 'Non-Quality_NonFunctional'
    ]

    all_data = {}
    for llm in all_metrics:
        avg_data = {}
        for method in methods:
            avg_metrics = {
                cls: {'P': [], 'R': [], 'F1': [], 'F2': []} for cls in classifications
            }
            for run_metrics in all_metrics[llm][method]:
                for cls in classifications:
                    if cls in run_metrics:
                        avg_metrics[cls]['P'].append(run_metrics[cls]['precision'])
                        avg_metrics[cls]['R'].append(run_metrics[cls]['recall'])
                        avg_metrics[cls]['F1'].append(run_metrics[cls]['f1'])
                        avg_metrics[cls]['F2'].append(run_metrics[cls]['f2'])
            avg_data[method] = {
                cls: {
                    'P': sum(avg_metrics[cls]['P']) / len(avg_metrics[cls]['P']) if avg_metrics[cls]['P'] else 0.0,
                    'R': sum(avg_metrics[cls]['R']) / len(avg_metrics[cls]['R']) if avg_metrics[cls]['R'] else 0.0,
                    'F1': sum(avg_metrics[cls]['F1']) / len(avg_metrics[cls]['F1']) if avg_metrics[cls]['F1'] else 0.0,
                    'F2': sum(avg_metrics[cls]['F2']) / len(avg_metrics[cls]['F2']) if avg_metrics[cls]['F2'] else 0.0
                } for cls in classifications
            }
            # Compute average over all classifications
            avg_data[method]['Average'] = {
                'P': sum(avg_data[method][cls]['P'] for cls in classifications) / len(classifications),
                'R': sum(avg_data[method][cls]['R'] for cls in classifications) / len(classifications),
                'F1': sum(avg_data[method][cls]['F1'] for cls in classifications) / len(classifications),
                'F2': sum(avg_data[method][cls]['F2'] for cls in classifications) / len(classifications)
            }
        all_data[llm] = avg_data

    with open('all_summary_metrics_promise_refined.csv', 'w') as f:
        for llm in all_data:
            f.write(f"{llm}\n")
            f.write("Method,Average P,Average R,Average F1,Average F2,F F1,F F2,Q F1,Q F2,onlyF F1,onlyF F2,onlyQ F1,onlyQ F2\n")
            for method in methods:
                avg = all_data[llm][method]['Average']
                f_f1 = all_data[llm][method]['Functional']['F1']
                f_f2 = all_data[llm][method]['Functional']['F2']
                q_f1 = all_data[llm][method]['Quality']['F1']
                q_f2 = all_data[llm][method]['Quality']['F2']
                onlyf_f1 = all_data[llm][method]['Functional_NonQuality']['F1']
                onlyf_f2 = all_data[llm][method]['Functional_NonQuality']['F2']
                onlyq_f1 = all_data[llm][method]['Quality_NonFunctional']['F1']
                onlyq_f2 = all_data[llm][method]['Quality_NonFunctional']['F2']
                method_name = 'CoT w Few-shot' if method == 'COT_with_examples' else 'Zero-shot' if method == 'ZeroShot' else method
                f.write(f"{method_name},{avg['P']:.3f},{avg['R']:.3f},{avg['F1']:.3f},{avg['F2']:.3f},"
                        f"{f_f1:.3f},{f_f2:.3f},{q_f1:.3f},{q_f2:.3f},"
                        f"{onlyf_f1:.3f},{onlyf_f2:.3f},{onlyq_f1:.3f},{onlyq_f2:.3f}\n")
            f.write("\n")

    return pd.DataFrame()

def run_api_classification(api_client, prompts_list, labels, df_test, method_name, run_num, is_quality=False):
    """Run classification using API clients for Functional/Quality tasks"""
    logger.info(f"Running {method_name} classification with {api_client.name} for {'Quality' if is_quality else 'Functional'}")
    
    predictions = []
    num_voting_runs = 3
    start_time = time.time()
    
    for voting_run in range(num_voting_runs):
        try:
            # Use batch classification with smaller batch size for APIs
            batch_size = 5  # Conservative batch size for API rate limits
            api_outputs = api_client.batch_classify(prompts_list, batch_size=batch_size)
            preds = mapping(api_outputs, labels)
            predictions.append(preds)
            
            # Check variance to potentially reduce voting runs
            if len(predictions) >= 2 and check_prediction_variance(predictions):
                break
                
        except Exception as e:
            logger.error(f"Error in {api_client.name} {method_name} run {voting_run + 1}: {str(e)}")
            # Add default predictions if there's an error
            predictions.append(['0'] * len(prompts_list))
    
    end_time = time.time()
    running_time = end_time - start_time
    
    voted_preds = majority_vote(predictions)
    metrics = compute_classification_metrics(
        df_test['label'].map({'1': 1, '0': 0}),
        [int(p) for p in voted_preds],
        is_quality=is_quality
    )
    
    return metrics, voted_preds, running_time

def run_classification(df_uploaded, num_runs=3, base_batch_size=16):
    llms = [
        "Qwen/Qwen2-7B-Instruct",
        "tiiuae/Falcon3-7B-Instruct",
        "ibm-granite/granite-3.2-8b-instruct",
        "mistralai/Ministral-8B-Instruct-2410",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]

    # Initialize API clients
    api_clients = initialize_api_clients()
    api_llms = list(api_clients.keys())
    
    all_llm_results = {llm: {method: [] for method in ['ZeroShot', 'FewShot', 'COT', 'COT_with_examples']} for llm in llms + api_llms}
    timing_data = []

    labels_functional = {
        'Non': '0',
        'Non-Functional': '0',
        'Functional': '1',
        '0': '0',
        '1': '1'
    }
    labels_quality = {
        'Non': '0',
        'Non-Quality': '0',
        'Quality': '1',
        '0': '0',
        '1': '1'
    }
    labels_fnq = {'Functional_NonQuality': '1', 'Non-Functional_NonQuality': '0', '0': '0', '1': '1'}
    labels_qnf = {'Quality_NonFunctional': '1', 'Non-Quality_NonFunctional': '0', '0': '0', '1': '1'}

    initial_descriptions_dict = {
        'Functional': """Requirements defining essential system functions, services, or behaviors under specific conditions. Specifies actions, operations, or transformations, focusing on inputs, outputs, and behavioral relationships. Excludes implementation constraints like performance or security.""",
        'Non-Functional': """Requirements addressing implementation constraints like performance, security, or usability, not essential system functions. Excludes specifications of actions, operations, or input-output relationships tied to system behavior.""",
        'Quality': """Requirements expressing how well a system executes functions, addressing attributes like performance, usability, reliability, or security. Includes Functional Suitability, Reliability, Performance Efficiency, Usability, Maintainability, Security, Compatibility, and Portability.""",
        'Non-Quality': """Requirements not expressing how well a system executes functions, excluding attributes like performance, usability, reliability, or security. Focuses on specific functional behaviors or implementation details without addressing global system properties."""
    }

    logger.info(f"Ground truth label counts:\n{df_uploaded.groupby(['Functional', 'Quality']).size().to_string()}")

    # Run local model classifications
    for llm in llms:
        logger.info(f"Processing local LLM: {llm}")
        
        pipeline, terminators = initialize_pipeline(llm)
        if pipeline is None:
            continue

        batch_size = get_dynamic_batch_size(llm, torch.cuda.mem_get_info()[0], base_batch_size)

        for run in range(num_runs):
            logger.info(f"Run {run+1}/{num_runs}")

            functional_examples, quality_examples, df_functional_train, df_functional_test, df_quality_train, df_quality_test = prepare_data(df_uploaded)

            initial_functional_examples = (
                df_functional_train.groupby('label', group_keys=False)
                .apply(lambda x: x.sample(min(4, len(x)), random_state=42 + run))
                .reset_index(drop=True)
                [['text', 'label']]
            )
            initial_quality_examples = (
                df_quality_train.groupby('label', group_keys=False)
                .apply(lambda x: x.sample(min(4, len(x)), random_state=42 + run))
                .reset_index(drop=True)
                [['text', 'label']]
            )

            logger.info(f"Functional examples: {initial_functional_examples.to_dict('records')}")
            logger.info(f"Quality examples: {initial_quality_examples.to_dict('records')}")

            class_examples_functional = initial_functional_examples.set_index('text')['label'].to_dict()
            class_examples_functional = {k: str(v) for k, v in class_examples_functional.items()}
            class_examples_quality = initial_quality_examples.set_index('text')['label'].to_dict()
            class_examples_quality = {k: str(v) for k, v in class_examples_quality.items()}

            zero_shot_prompts_f, few_shot_prompts_f, cot_prompts_f, cot_with_examples_prompts_f = generate_prompts(
                df_functional_test,
                class_examples_functional,
                {'Functional': initial_descriptions_dict['Functional'], 'Non-Functional': initial_descriptions_dict['Non-Functional']}
            )
            zero_shot_prompts_q, few_shot_prompts_q, cot_prompts_q, cot_with_examples_prompts_q = generate_prompts(
                df_quality_test,
                class_examples_quality,
                {'Quality': initial_descriptions_dict['Quality'], 'Non-Quality': initial_descriptions_dict['Non-Quality']},
                is_quality=True
            )

            methods_f = {
                'ZeroShot': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in zero_shot_prompts_f],
                'FewShot': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in few_shot_prompts_f],
                'COT': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in cot_prompts_f],
                'COT_with_examples': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in cot_with_examples_prompts_f]
            }
            methods_q = {
                'ZeroShot': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in zero_shot_prompts_q],
                'FewShot': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in few_shot_prompts_q],
                'COT': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in cot_prompts_q],
                'COT_with_examples': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in cot_with_examples_prompts_q]
            }

            for method in methods_f:
                predictions_f = []
                predictions_q = []
                num_voting_runs = 3
                start_time = time.time()
                for _ in range(num_voting_runs):
                    dataset_f = Dataset.from_dict({"prompt": methods_f[method]})
                    outputs_f = pipeline(
                        dataset_f["prompt"],
                        max_new_tokens=1024,
                        eos_token_id=terminators[0],
                        batch_size=batch_size,
                        pad_token_id=pipeline.tokenizer.pad_token_id
                    )
                    preds_f = mapping(remove_assistant_prefix(outputs_f), labels_functional)
                    predictions_f.append(preds_f)

                    dataset_q = Dataset.from_dict({"prompt": methods_q[method]})
                    outputs_q = pipeline(
                        dataset_q["prompt"],
                        max_new_tokens=1024,
                        eos_token_id=terminators[0],
                        batch_size=batch_size,
                        pad_token_id=pipeline.tokenizer.pad_token_id
                    )
                    preds_q = mapping(remove_assistant_prefix(outputs_q), labels_quality)
                    predictions_q.append(preds_q)

                    if len(predictions_f) >= 2 and check_prediction_variance(predictions_f) and check_prediction_variance(predictions_q):
                        break
                end_time = time.time()
                running_time = end_time - start_time

                voted_preds_f = majority_vote(predictions_f)
                voted_preds_q = majority_vote(predictions_q)

                # Derive predictions for Functional_NonQuality_vs_All using LLM predictions
                voted_preds_fnq = [
                    '1' if f == '1' and q == '0' else '0'
                    for f, q in zip(voted_preds_f, voted_preds_q)
                ]
                # Derive ground truth for Functional_NonQuality_vs_All using input data
                ground_truth_fnq = [
                    '1' if f == '1' and q == '0' else '0'
                    for f, q in zip(df_functional_test['label'], df_quality_test['label'])
                ]

                # Derive predictions for Quality_NonFunctional_vs_All using LLM predictions
                voted_preds_qnf = [
                    '1' if q == '1' and f == '0' else '0'
                    for f, q in zip(voted_preds_f, voted_preds_q)
                ]
                # Derive ground truth for Quality_NonFunctional_vs_All using input data
                ground_truth_qnf = [
                    '1' if q == '1' and f == '0' else '0'
                    for f, q in zip(df_functional_test['label'], df_quality_test['label'])
                ]

                # Compute metrics using input data ground truth for Functional and Quality
                metrics_f = compute_classification_metrics(
                    df_functional_test['label'].map({'1': 1, '0': 0}),
                    [int(p) for p in voted_preds_f],
                    is_quality=False
                )
                metrics_q = compute_classification_metrics(
                    df_quality_test['label'].map({'1': 1, '0': 0}),
                    [int(p) for p in voted_preds_q],
                    is_quality=True
                )
                metrics_fnq = compute_classification_metrics(
                    [int(g) for g in ground_truth_fnq],
                    [int(p) for p in voted_preds_fnq],
                    is_quality=False
                )
                metrics_qnf = compute_classification_metrics(
                    [int(g) for g in ground_truth_qnf],
                    [int(p) for p in voted_preds_qnf],
                    is_quality=False
                )

                # Rename metrics for Functional_NonQuality and Quality_NonFunctional
                metrics_fnq = {
                    'Functional_NonQuality': metrics_fnq['Functional'],
                    'Non-Functional_NonQuality': metrics_fnq['Non-Functional'],
                    'avg': metrics_fnq['avg']
                }
                metrics_qnf = {
                    'Quality_NonFunctional': metrics_qnf['Functional'],
                    'Non-Quality_NonFunctional': metrics_qnf['Non-Functional'],
                    'avg': metrics_qnf['avg']
                }

                # Combine metrics for storage
                combined_metrics = {
                    'Functional': metrics_f['Functional'],
                    'Non-Functional': metrics_f['Non-Functional'],
                    'Quality': metrics_q['Quality'],
                    'Non-Quality': metrics_q['Non-Quality'],
                    'Functional_NonQuality': metrics_fnq['Functional_NonQuality'],
                    'Non-Functional_NonQuality': metrics_fnq['Non-Functional_NonQuality'],
                    'Quality_NonFunctional': metrics_qnf['Quality_NonFunctional'],
                    'Non-Quality_NonFunctional': metrics_qnf['Non-Quality_NonFunctional'],
                    'avg': {
                        'precision': (metrics_f['avg']['precision'] + metrics_q['avg']['precision'] + 
                                     metrics_fnq['avg']['precision'] + metrics_qnf['avg']['precision']) / 4,
                        'recall': (metrics_f['avg']['recall'] + metrics_q['avg']['recall'] + 
                                  metrics_fnq['avg']['recall'] + metrics_qnf['avg']['recall']) / 4,
                        'f1': (metrics_f['avg']['f1'] + metrics_q['avg']['f1'] + 
                              metrics_fnq['avg']['f1'] + metrics_qnf['avg']['f1']) / 4,
                        'f2': (metrics_f['avg']['f2'] + metrics_q['avg']['f2'] + 
                              metrics_fnq['avg']['f2'] + metrics_qnf['avg']['f2']) / 4
                    }
                }
                all_llm_results[llm][method].append(combined_metrics)

                # Record timing for Functional and Quality tasks
                timing_data.append({
                    'LLM': llm,
                    'Run': run + 1,
                    'Task': 'Functional vs Non-Functional Classification',
                    'Method': method,
                    'Running_Time': running_time / 2,
                    'Type': 'local'
                })
                timing_data.append({
                    'LLM': llm,
                    'Run': run + 1,
                    'Task': 'Quality vs Non-Quality Classification',
                    'Method': method,
                    'Running_Time': running_time / 2,
                    'Type': 'local'
                })
                timing_data.append({
                    'LLM': llm,
                    'Run': run + 1,
                    'Task': 'Functional_NonQuality vs All Classification',
                    'Method': method,
                    'Running_Time': 0.0,
                    'Type': 'local'
                })
                timing_data.append({
                    'LLM': llm,
                    'Run': run + 1,
                    'Task': 'Quality_NonFunctional vs All Classification',
                    'Method': method,
                    'Running_Time': 0.0,
                    'Type': 'local'
                })

        logger.info(f"Flushing GPU memory after processing {llm}")
        del pipeline
        torch.cuda.empty_cache()

    # Run API classifications
    for api_name, api_client in api_clients.items():
        logger.info(f"Processing API: {api_name}")
        
        for run in range(num_runs):
            logger.info(f"Run {run+1}/{num_runs} for {api_name}")

            functional_examples, quality_examples, df_functional_train, df_functional_test, df_quality_train, df_quality_test = prepare_data(df_uploaded)

            initial_functional_examples = (
                df_functional_train.groupby('label', group_keys=False)
                .apply(lambda x: x.sample(min(4, len(x)), random_state=42 + run))
                .reset_index(drop=True)
                [['text', 'label']]
            )
            initial_quality_examples = (
                df_quality_train.groupby('label', group_keys=False)
                .apply(lambda x: x.sample(min(4, len(x)), random_state=42 + run))
                .reset_index(drop=True)
                [['text', 'label']]
            )

            class_examples_functional = initial_functional_examples.set_index('text')['label'].to_dict()
            class_examples_functional = {k: str(v) for k, v in class_examples_functional.items()}
            class_examples_quality = initial_quality_examples.set_index('text')['label'].to_dict()
            class_examples_quality = {k: str(v) for k, v in class_examples_quality.items()}

            # Generate prompts (same format as for local models)
            zero_shot_prompts_f, few_shot_prompts_f, cot_prompts_f, cot_with_examples_prompts_f = generate_prompts(
                df_functional_test,
                class_examples_functional,
                {'Functional': initial_descriptions_dict['Functional'], 'Non-Functional': initial_descriptions_dict['Non-Functional']}
            )
            zero_shot_prompts_q, few_shot_prompts_q, cot_prompts_q, cot_with_examples_prompts_q = generate_prompts(
                df_quality_test,
                class_examples_quality,
                {'Quality': initial_descriptions_dict['Quality'], 'Non-Quality': initial_descriptions_dict['Non-Quality']},
                is_quality=True
            )

            methods_f = {
                'ZeroShot': zero_shot_prompts_f,
                'FewShot': few_shot_prompts_f,
                'COT': cot_prompts_f,
                'COT_with_examples': cot_with_examples_prompts_f
            }
            methods_q = {
                'ZeroShot': zero_shot_prompts_q,
                'FewShot': few_shot_prompts_q,
                'COT': cot_prompts_q,
                'COT_with_examples': cot_with_examples_prompts_q
            }

            for method in methods_f:
                start_time = time.time()
                
                # Run Functional classification
                metrics_f, voted_preds_f, time_f = run_api_classification(
                    api_client, methods_f[method], labels_functional, df_functional_test, method, run, is_quality=False
                )
                
                # Run Quality classification
                metrics_q, voted_preds_q, time_q = run_api_classification(
                    api_client, methods_q[method], labels_quality, df_quality_test, method, run, is_quality=True
                )
                
                end_time = time.time()
                total_running_time = end_time - start_time

                # Derive predictions for Functional_NonQuality_vs_All using LLM predictions
                voted_preds_fnq = [
                    '1' if f == '1' and q == '0' else '0'
                    for f, q in zip(voted_preds_f, voted_preds_q)
                ]
                # Derive ground truth for Functional_NonQuality_vs_All using input data
                ground_truth_fnq = [
                    '1' if f == '1' and q == '0' else '0'
                    for f, q in zip(df_functional_test['label'], df_quality_test['label'])
                ]

                # Derive predictions for Quality_NonFunctional_vs_All using LLM predictions
                voted_preds_qnf = [
                    '1' if q == '1' and f == '0' else '0'
                    for f, q in zip(voted_preds_f, voted_preds_q)
                ]
                # Derive ground truth for Quality_NonFunctional_vs_All using input data
                ground_truth_qnf = [
                    '1' if q == '1' and f == '0' else '0'
                    for f, q in zip(df_functional_test['label'], df_quality_test['label'])
                ]

                # Compute metrics for derived classifications
                metrics_fnq = compute_classification_metrics(
                    [int(g) for g in ground_truth_fnq],
                    [int(p) for p in voted_preds_fnq],
                    is_quality=False
                )
                metrics_qnf = compute_classification_metrics(
                    [int(g) for g in ground_truth_qnf],
                    [int(p) for p in voted_preds_qnf],
                    is_quality=False
                )

                # Rename metrics for Functional_NonQuality and Quality_NonFunctional
                metrics_fnq = {
                    'Functional_NonQuality': metrics_fnq['Functional'],
                    'Non-Functional_NonQuality': metrics_fnq['Non-Functional'],
                    'avg': metrics_fnq['avg']
                }
                metrics_qnf = {
                    'Quality_NonFunctional': metrics_qnf['Functional'],
                    'Non-Quality_NonFunctional': metrics_qnf['Non-Functional'],
                    'avg': metrics_qnf['avg']
                }

                # Combine metrics for storage
                combined_metrics = {
                    'Functional': metrics_f['Functional'],
                    'Non-Functional': metrics_f['Non-Functional'],
                    'Quality': metrics_q['Quality'],
                    'Non-Quality': metrics_q['Non-Quality'],
                    'Functional_NonQuality': metrics_fnq['Functional_NonQuality'],
                    'Non-Functional_NonQuality': metrics_fnq['Non-Functional_NonQuality'],
                    'Quality_NonFunctional': metrics_qnf['Quality_NonFunctional'],
                    'Non-Quality_NonFunctional': metrics_qnf['Non-Quality_NonFunctional'],
                    'avg': {
                        'precision': (metrics_f['avg']['precision'] + metrics_q['avg']['precision'] + 
                                     metrics_fnq['avg']['precision'] + metrics_qnf['avg']['precision']) / 4,
                        'recall': (metrics_f['avg']['recall'] + metrics_q['avg']['recall'] + 
                                  metrics_fnq['avg']['recall'] + metrics_qnf['avg']['recall']) / 4,
                        'f1': (metrics_f['avg']['f1'] + metrics_q['avg']['f1'] + 
                              metrics_fnq['avg']['f1'] + metrics_qnf['avg']['f1']) / 4,
                        'f2': (metrics_f['avg']['f2'] + metrics_q['avg']['f2'] + 
                              metrics_fnq['avg']['f2'] + metrics_qnf['avg']['f2']) / 4
                    }
                }
                all_llm_results[api_name][method].append(combined_metrics)

                # Record timing for API tasks
                timing_data.append({
                    'LLM': api_name,
                    'Run': run + 1,
                    'Task': 'Functional vs Non-Functional Classification',
                    'Method': method,
                    'Running_Time': time_f,
                    'Type': 'api'
                })
                timing_data.append({
                    'LLM': api_name,
                    'Run': run + 1,
                    'Task': 'Quality vs Non-Quality Classification',
                    'Method': method,
                    'Running_Time': time_q,
                    'Type': 'api'
                })
                timing_data.append({
                    'LLM': api_name,
                    'Run': run + 1,
                    'Task': 'Functional_NonQuality vs All Classification',
                    'Method': method,
                    'Running_Time': 0.0,
                    'Type': 'api'
                })
                timing_data.append({
                    'LLM': api_name,
                    'Run': run + 1,
                    'Task': 'Quality_NonFunctional vs All Classification',
                    'Method': method,
                    'Running_Time': 0.0,
                    'Type': 'api'
                })

    timing_df = pd.DataFrame(timing_data)
    timing_df.to_csv('llm_task_timing_promise_refined.csv', index=False)
    logger.info("Timing data saved to 'llm_task_timing_promise_refined.csv'.")

    generate_summary_table_custom(all_llm_results, num_runs)
    logger.info("Summary metrics saved to 'all_summary_metrics_promise_refined.csv'.")

    return all_llm_results

if __name__ == "__main__":
    file_path = './promise-reclass.csv'
    
    # Load the CSV directly with robust parsing
    try:
        df_uploaded = pd.read_csv(file_path, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        raise
    
    df_uploaded['Functional'] = df_uploaded['IsFunctional'].apply(lambda x: '1' if x == 1 else '0')
    df_uploaded['Quality'] = df_uploaded['IsQuality'].apply(lambda x: '1' if x == 1 else '0')
    df_uploaded = df_uploaded.rename(columns={'RequirementText': 'text'})
    
    # Log dataset statistics
    logger.info(f"Dataset size: {len(df_uploaded)}")
    logger.info(f"Duplicate texts: {df_uploaded['text'].duplicated().sum()}")
    logger.info(f"Functional label distribution:\n{df_uploaded['Functional'].value_counts().to_string()}")
    logger.info(f"Quality label distribution:\n{df_uploaded['Quality'].value_counts().to_string()}")
    
    run_classification(df_uploaded, num_runs=3, base_batch_size=16)