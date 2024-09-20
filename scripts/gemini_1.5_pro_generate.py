# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python (myenv)
#     language: python
#     name: myenv
# ---

# +
import json
import argparse
import vertexai
from transformers import AutoTokenizer
from vertexai.generative_models import GenerativeModel, Part

def load_data_from_local(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_data_to_local(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved locally at {file_path}")

def truncate_texts(real_text, sample_text):
    real_words = real_text.split()
    sample_words = sample_text.split()
    max_length = min(len(real_words), len(sample_words))
    return ' '.join(real_words[:max_length]), ' '.join(sample_words[:max_length])

'''
def generate_samples(data, model):
    generated_data = {'original': [], 'sampled': []}
    tokenizer = AutoTokenizer.from_pretrained('t5-small')  # Initialize the tokenizer
    
    for item in data:
        tokens = tokenizer.tokenize(item['document'])
        prefix_tokens = tokens[:30]
        prefix = tokenizer.convert_tokens_to_string(prefix_tokens)
        full_prompt = f"You are an Olympic News writer. Please write an article with about 150 words starting exactly with '{prefix}'."

        response = model.generate_content([Part.from_text(full_prompt)])
        truncated_real, truncated_sample = truncate_texts(item['document'], response.text)
        generated_data['original'].append(truncated_real)
        generated_data['sampled'].append(truncated_sample)
        
    return generated_data
'''
def generate_samples(data, model):
    generated_data = {'original': [], 'sampled': []}
    tokenizer = AutoTokenizer.from_pretrained('t5-small')  # Initialize the tokenizer
    
    for item in data:
        try:
            input_ids = tokenizer.encode(item['document'], add_special_tokens=False, truncation=True, max_length=30)
            prefix = tokenizer.decode(input_ids, skip_special_tokens=True)
            full_prompt = f"You are a News writer. Please write an article with about 150 words starting exactly with '{prefix}'."


            response = model.generate_content([Part.from_text(full_prompt)])
            truncated_original, truncated_sampled = truncate_texts(item['document'], response.text)
            generated_data['original'].append(truncated_original)
            generated_data['sampled'].append(truncated_sampled)
        except Exception as e:  # 捕获由于安全过滤器或其他原因引发的异常
            print(f"Skipping item due to safety filters or other issues: {e}")
            continue

    return generated_data
def main(args):
    vertexai.init(project=args.project_id, location=args.location_name)
    model = GenerativeModel(model_name=args.model_name)

    data = load_data_from_local(args.input_file)
    generated_data = generate_samples(data, model)
    save_data_to_local(args.output_file, generated_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate samples with a pre-trained model.")
    parser.add_argument('--input_file', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--project_id', type=str, required=True, help='GCP project ID')
    parser.add_argument('--location_name', type=str, default='us-central1', help='GCP location')
    parser.add_argument('--model_name', type=str, required=True, help='Vertex AI model name')
    
    args = parser.parse_args()
    main(args)

