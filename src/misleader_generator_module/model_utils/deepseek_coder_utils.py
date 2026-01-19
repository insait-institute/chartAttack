from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import logging

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

def read_context(file_name: str):
    context = open(file_name, 'r').read()
    return context

def extract_code_snippet(text):
    """
    Extracts the Python code snippet from a given text enclosed by triple backticks (```python ... ```).

    Args:
        text (str): The input text containing the Python code snippet.

    Returns:
        str: The extracted Python code snippet, or None if no snippet is found.
    """
    # Regular expression to match code enclosed in triple backticks
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def set_oracle_messages(query: str, gold_answer:str, annotations: dict, few_shot_examples: list, context: str):
    messages = []
    system_dict = {"role": "system", "content": context}
    messages.append(system_dict)
    for example in few_shot_examples:
        example_question = example["question"]
        example_annotations = example["annotations"]
        example_output = example["output"]
        example_gold_answer = example["gold"]
        user_dict = {"role": "user", "content": f"Annotations:\n{json.dumps(example_annotations)}\nQuestion:{example_question}\nCorrect answer:{example_gold_answer}"}
        assistant_dict = {"role": "assistant", "content": f"```python\n{json.dumps(example_output, separators=(',',':'))}\n```"}
        messages.append(user_dict)
        messages.append(assistant_dict)
    user_dict = {"role": "user", "content": f"Annotations:\n{json.dumps(annotations)}\nQuestion:{query}\nCorrect answer:{gold_answer}"}
    messages.append(user_dict)
    return messages

def read_json(file_name: str):
    with open(file_name, 'r') as file:
        json_data = json.load(file)
    return json_data

def get_annotations(imgname: str, chart_type: str, partition, dataset: str):
    annotations_path = f'/ukp-storage-1/ortiz/datasets/{dataset}/clean_{partition}/enhanced_simplified_annotations_v2/{chart_type}/{imgname}.json'
    annotations = read_json(annotations_path)
    return annotations

def deepseek_coder_inference(args, questions, context_path):
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16 ,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, trust_remote_code=True)
    context = read_context(context_path)
    processed_responses = []
    for item in questions:
        query = item["question"]
        gold_answer = item["gold"]
        annotations = get_annotations(item["imgname"], args.chart_type, args.partition, args.dataset)
        few_shot_examples = item["few_shot_examples"] if args.zero_shot == False else []
        messages = set_oracle_messages(query, gold_answer, annotations, few_shot_examples, context)
        inputs = tokenizer.apply_chat_template(messages,add_generation_prompt=True, return_tensors="pt").to(model.device)
        
        outputs = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        response = extract_code_snippet(response)
        processed_response = {
            "imgname": item["imgname"],
            "question":item["question"],
            "gold": item["gold"],
            "question_class": item["question_class"],
            "possible_misleading_techniques": item["possible_misleading_techniques"],
            "response": response
        }
        logging.debug(f'{processed_response}')
        processed_responses.append(processed_response)
    
    return processed_responses

