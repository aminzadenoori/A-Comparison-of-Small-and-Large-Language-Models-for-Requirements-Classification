import os
import re
import pandas as pd
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import classification_report, precision_recall_fscore_support, fbeta_score
from sklearn.model_selection import train_test_split
import logging
import numpy as np
from collections import deque
import time
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
                    batch_results.append("0")  # Default to non-security on error
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
                model="grok-4",  # Adjust model name as needed
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
        self.model = "claude-4-sonnet-20240229"  # Adjust model as needed
    
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
        
        if pipeline.tokenizer is None:
            logger.error(f"Tokenizer for {model_name} is None")
            return None, None
        
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

def generate_prompts(df, class_examples, class_explanations):
    """Generate prompts using class examples and explanations."""
    explanation_str = "\n".join([f'{label}: {explanation}' for label, explanation in class_explanations.items()])
    example_str = "\n".join([f'"{text}" --> {label}' for text, label in class_examples.items()])
    
    zero_shot_prompts = generate_all_prompts(df['text'].tolist(), explanations=explanation_str)
    few_shot_prompts = generate_all_prompts(df['text'].tolist(), examples=example_str, explanations=explanation_str)
    cot_prompts = generate_all_prompts(df['text'].tolist(), explanations=explanation_str, cot=True)
    cot_with_examples_prompts = generate_all_prompts(df['text'].tolist(), examples=example_str, explanations=explanation_str, cot=True)
    
    return zero_shot_prompts, few_shot_prompts, cot_prompts, cot_with_examples_prompts

def generate_messages(text, examples=None, explanations=None, cot=False, is_quality=False):
    system_content = f"""
As an expert system for classifying software requirements, your task is to carefully analyze each given requirement and categorize it into one of the following two categories:
"1": Security
"0": Non-Security
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

def sample_mixed_examples(df, correct_col, label_col, total_samples_per_class=4, random_state=None):
    correct_df = df[df[correct_col]]
    incorrect_df = df[~df[correct_col]]
    labels = df[label_col].unique()
    n_samples_per_label = total_samples_per_class // 2
    sampled_dfs = []
    for label in labels:
        correct_label_df = correct_df[correct_df[label_col] == label]
        n_correct = min(n_samples_per_label, len(correct_label_df))
        if n_correct > 0:
            sampled_correct = correct_label_df.sample(n_correct, random_state=random_state)
            sampled_dfs.append(sampled_correct)
        incorrect_label_df = incorrect_df[incorrect_df[label_col] == label]
        n_incorrect = min(n_samples_per_label, len(incorrect_label_df))
        if n_incorrect > 0:
            sampled_incorrect = incorrect_label_df.sample(n_incorrect, random_state=random_state)
            sampled_dfs.append(sampled_incorrect)
    return pd.concat(sampled_dfs).reset_index(drop=True) if sampled_dfs else pd.DataFrame(columns=df.columns)

def compute_classification_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['Non-Security', 'Security'], output_dict=True, zero_division=0)
    metrics = {
        'Security': {
            'precision': report['Security']['precision'],
            'recall': report['Security']['recall'],
            'f1': report['Security']['f1-score'],
            'f2': fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
        },
        'Non-Security': {
            'precision': report['Non-Security']['precision'],
            'recall': report['Non-Security']['recall'],
            'f1': report['Non-Security']['f1-score'],
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
            voted.append('0')
    return voted

def prepare_data(df):
    df_train, df_test = train_test_split(
        df, test_size=0.7, random_state=42, stratify=df['label']
    )
    security_examples = df_train.drop_duplicates()[['text', 'label']]
    return security_examples, df_train, df_test

def generate_summary_table_custom(all_metrics, num_runs):
    methods = ['ZeroShot', 'FewShot', 'COT', 'COT_with_examples']
    classifications = ['Security', 'Non-Security']
    metrics_list = ['precision', 'recall', 'f1', 'f2']

    all_data = {}
    for llm in all_metrics:
        avg_data = {}
        for method in methods:
            avg_p = []
            avg_r = []
            avg_f1 = []
            avg_f2 = []
            for classification in classifications:
                values_p = [run_metrics[classification]['precision'] for run_metrics in all_metrics[llm][method] if run_metrics]
                values_r = [run_metrics[classification]['recall'] for run_metrics in all_metrics[llm][method] if run_metrics]
                values_f1 = [run_metrics[classification]['f1'] for run_metrics in all_metrics[llm][method] if run_metrics]
                values_f2 = [run_metrics[classification]['f2'] for run_metrics in all_metrics[llm][method] if run_metrics]
                avg_p.extend(values_p)
                avg_r.extend(values_r)
                avg_f1.extend(values_f1)
                avg_f2.extend(values_f2)
            avg_data[method] = {
                'P': sum(avg_p) / len(avg_p) if avg_p else 0.0,
                'R': sum(avg_r) / len(avg_r) if avg_r else 0.0,
                'F1': sum(avg_f1) / len(avg_f1) if avg_f1 else 0.0,
                'F2': sum(avg_f2) / len(avg_f2) if avg_f2 else 0.0
            }

        best_data = {}
        for method in methods:
            best_data[method] = {}
            for classification in classifications:
                values_f1 = [run_metrics[classification]['f1'] for run_metrics in all_metrics[llm][method] if run_metrics]
                best_f1 = max(values_f1, default=0.0)
                for run_metrics in all_metrics[llm][method]:
                    if run_metrics and run_metrics[classification]['f1'] == best_f1:
                        best_data[method][classification] = {
                            'F1': best_f1,
                            'F2': run_metrics[classification]['f2']
                        }
                        break
                if classification not in best_data[method]:
                    best_data[method][classification] = {'F1': 0.0, 'F2': 0.0}
        all_data[llm] = {'avg': avg_data, 'best': best_data}

    with open('all_summary_metrics_security.csv', 'w') as f:
        for llm in all_data:
            f.write(f"{llm}\n")
            f.write("\tAverage\t\t\t\tSecurity\t\tNon-Security\n")
            f.write("\tP\tR\tF1\tF2\tF1\tF2\tF1\tF2\n")
            for method in methods:
                avg = all_data[llm]['avg'][method]
                best = all_data[llm]['best'][method]
                method_name = 'CoT w Few-shot' if method == 'COT_with_examples' else 'Zero-shot' if method == 'ZeroShot' else method
                f.write(f"{method_name}\t{avg['P']:.3f}\t{avg['R']:.3f}\t{avg['F1']:.3f}\t{avg['F2']:.3f}\t"
                        f"{best['Security']['F1']:.3f}\t{best['Security']['F2']:.3f}\t"
                        f"{best['Non-Security']['F1']:.3f}\t{best['Non-Security']['F2']:.3f}\n")
            f.write("\n\n")

    return pd.DataFrame()

def run_api_classification(api_client, prompts_list, labels_security, df_test, method_name, run_num):
    """Run classification using API clients"""
    logger.info(f"Running {method_name} classification with {api_client.name}")
    
    predictions = []
    num_voting_runs = 3
    start_time = time.time()
    
    for voting_run in range(num_voting_runs):
        try:
            # Use batch classification with smaller batch size for APIs
            batch_size = 5  # Conservative batch size for API rate limits
            api_outputs = api_client.batch_classify(prompts_list, batch_size=batch_size)
            preds = mapping(api_outputs, labels_security)
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
    metrics = compute_classification_metrics(df_test['label'], [int(p) for p in voted_preds])
    
    return metrics, running_time

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

    labels_security = {
        'Non': '0',
        'Non-Security': '0',
        'Security': '1',
        '0': '0',
        '1': '1'
    }

    initial_descriptions_dict = {
        'Security': """Requirements ensuring protection of data, systems, or users against unauthorized access, breaches, or threats. Covers encryption, authentication, access controls, and threat mitigation. Excludes functional, performance, or usability features not tied to security.""",
        'Non-Security': """Requirements defining system functionality, performance, or usability, unrelated to data or system protection. Includes user interfaces, operational features, and efficiency metrics. Excludes encryption, authentication, or access control mechanisms."""
    }
    initial_descriptions = "\n".join([f'{label}: {explanation}' for label, explanation in initial_descriptions_dict.items()])

    # Run local model classifications
    for llm in llms:
        logger.info(f"Processing local LLM: {llm}")
        
        pipeline, terminators = initialize_pipeline(llm)
        if pipeline is None:
            continue

        batch_size = get_dynamic_batch_size(llm, torch.cuda.mem_get_info()[0], base_batch_size)

        for run in range(num_runs):
            logger.info(f"Run {run+1}/{num_runs}")

            security_examples, df_train, df_test = prepare_data(df_uploaded)

            initial_security_examples = (
                df_train.groupby('label', group_keys=False)
                .apply(lambda x: x.sample(min(4, len(x)), random_state=42 + run))
                .reset_index(drop=True)
                [['text', 'label']]
            )
            class_examples_security = initial_security_examples.set_index('text')['label'].to_dict()
            class_examples_security = {k: str(v) for k, v in class_examples_security.items()}

            zero_shot_prompts_s, few_shot_prompts_s, cot_prompts_s, cot_with_examples_prompts_s = generate_prompts(
                df_test,
                class_examples_security,
                initial_descriptions_dict
            )
            methods_s = {
                'ZeroShot': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in zero_shot_prompts_s],
                'FewShot': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in few_shot_prompts_s],
                'COT': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in cot_prompts_s],
                'COT_with_examples': [pipeline.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in cot_with_examples_prompts_s]
            }

            for method, prompts in methods_s.items():
                dataset_s = Dataset.from_dict({"prompt": prompts})
                predictions = []
                num_voting_runs = 3
                start_time = time.time()
                for _ in range(num_voting_runs):
                    outputs_s = pipeline(dataset_s["prompt"], max_new_tokens=1024, eos_token_id=terminators[0], batch_size=batch_size)
                    preds_s = mapping(remove_assistant_prefix(outputs_s), labels_security)
                    predictions.append(preds_s)
                    if len(predictions) >= 2 and check_prediction_variance(predictions):
                        break
                end_time = time.time()
                running_time = end_time - start_time

                voted_preds = majority_vote(predictions)
                metrics = compute_classification_metrics(df_test['label'], [int(p) for p in voted_preds])
                all_llm_results[llm][method].append(metrics)

                timing_data.append({
                    'LLM': llm,
                    'Run': run + 1,
                    'Method': method,
                    'Running_Time': running_time,
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

            security_examples, df_train, df_test = prepare_data(df_uploaded)

            initial_security_examples = (
                df_train.groupby('label', group_keys=False)
                .apply(lambda x: x.sample(min(4, len(x)), random_state=42 + run))
                .reset_index(drop=True)
                [['text', 'label']]
            )
            class_examples_security = initial_security_examples.set_index('text')['label'].to_dict()
            class_examples_security = {k: str(v) for k, v in class_examples_security.items()}

            # Generate prompts (same format as for local models)
            zero_shot_prompts_s, few_shot_prompts_s, cot_prompts_s, cot_with_examples_prompts_s = generate_prompts(
                df_test,
                class_examples_security,
                initial_descriptions_dict
            )

            methods_s = {
                'ZeroShot': zero_shot_prompts_s,
                'FewShot': few_shot_prompts_s,
                'COT': cot_prompts_s,
                'COT_with_examples': cot_with_examples_prompts_s
            }

            for method, prompts in methods_s.items():
                metrics, running_time = run_api_classification(
                    api_client, prompts, labels_security, df_test, method, run
                )
                all_llm_results[api_name][method].append(metrics)

                timing_data.append({
                    'LLM': api_name,
                    'Run': run + 1,
                    'Method': method,
                    'Running_Time': running_time,
                    'Type': 'api'
                })

    timing_df = pd.DataFrame(timing_data)
    timing_df.to_csv('llm_task_timing.csv', index=False)
    logger.info("Timing data saved to 'llm_task_timing.csv'.")

    generate_summary_table_custom(all_llm_results, num_runs)
    logger.info("Summary metrics saved to 'all_summary_metrics_security.csv'.")

    return all_llm_results

if __name__ == "__main__":
    file_path = './CPN.csv'
    df_uploaded = pd.read_csv(
        file_path,
        sep=';',
        header=None,
        names=['RequirementText', 'label'],
        encoding='cp1252'
    )

    df_uploaded['label'] = df_uploaded['label'].apply(lambda x: 1 if x.lower() == 'sec' else 0)
    df_uploaded = df_uploaded.rename(columns={'RequirementText': 'text'})
    
    run_classification(df_uploaded, num_runs=3, base_batch_size=16)
