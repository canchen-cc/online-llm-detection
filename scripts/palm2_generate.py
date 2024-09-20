import json
import argparse
import vertexai
from transformers import AutoTokenizer
from vertexai.preview.language_models import TextGenerationModel

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

def generate_samples(data, model, temperature, max_decode_steps, top_p, top_k):
    generated_data = {'original': [], 'sampled': []}
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    
    for item in data:
        # Tokenize with proper truncation and padding
        tokens = tokenizer.encode(item['document'], truncation=True, max_length=512, return_tensors="pt")
        if tokens.shape[1] > 30:
            prefix_tokens = tokens[:, :30]  # Only take the first 30 tokens
        else:
            prefix_tokens = tokens

        # Convert tokens to string for prefix
        prefix = tokenizer.decode(prefix_tokens[0], skip_special_tokens=True)
        full_prompt = f"You are a News writer. Please write an article with about 150 words starting exactly with '{prefix}'."

        try:
            # Assuming `model` is an instance of a model that has a `generate` method
            response = model.predict(
            full_prompt,
            temperature=temperature,
            max_output_tokens=max_decode_steps,
            top_k=top_k,
            top_p=top_p
            )

            truncated_real, truncated_sample = truncate_texts(item['document'], response.text)
            generated_data['original'].append(truncated_real)
            generated_data['sampled'].append(truncated_sample)
        except Exception as e:  # General exception handling
            print(f"Skipping item due to error: {e}")
            continue

    return generated_data

def main(args):
    vertexai.init(project=args.project_id, location=args.location_name)
    model = TextGenerationModel.from_pretrained(args.model_name)

    data = load_data_from_local(args.input_file)
    generated_data = generate_samples(data, model, args.temperature, args.max_decode_steps, args.top_p, args.top_k)
    save_data_to_local(args.output_file, generated_data)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate samples with a pre-trained model.")
    parser.add_argument('--input_file', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--project_id', type=str, required=True, help='GCP project ID')
    parser.add_argument('--location_name', type=str, default='us-central1', help='GCP location')
    parser.add_argument('--model_name', type=str, required=True, help='Vertex AI model name')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_decode_steps', type=int, default=200)
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--top_k', type=int, default=40)
    args = parser.parse_args()
    main(args)
