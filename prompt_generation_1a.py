import json
import os

# The exact instruction prompt for your LLM
INSTRUCTION = "Explain the answer to the following question as if I were a 5-year-old. Base your explanation strictly on the provided context. Provide ONLY your final explanation. Do not generate any extra documents, follow-up questions, or separators."

def load_data(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as file:
        return json.load(file)

def generate_experiment_1a_distractors(dataset):
    """
    Experiment 1A: Impact of Distracting Documents
    Layout: [ Instruction, N x S (Distractors), G (Gold), Question ]
    """
    # The different context lengths (number of distractor documents)
    distractor_counts = [0, 4, 9, 14, 19, 24, 29]
    
    # Create the output directory
    output_dir = "prompts/exp1a"
    os.makedirs(output_dir, exist_ok=True)

    for count in distractor_counts:
        case_prompts = []
        
        for entry in dataset:
            query_id = entry.get('query_id', 'unknown')
            question = entry.get('question', '')
            
            # 1. Extract the Gold text
            gold_data = entry.get('gold', [])
            gold_text = " ".join([d.get("text", "") for d in gold_data if isinstance(d, dict) and "text" in d])
            
            # 2. Extract the Distractors text (up to the current count)
            distractor_data = entry.get('distractors', [])
            S_docs = [d.get("text", "") for d in distractor_data if isinstance(d, dict) and "text" in d]
            
            # Slice the array to the required number of distractors
            S_docs = S_docs[:count]
            
            if len(S_docs) < count:
                print(f"Warning: Entry {query_id} only has {len(S_docs)} distractors, but {count} requested.")
            
            # 3. Construct layout: S distractor documents followed by the Gold document
            context_blocks = S_docs + [gold_text]
            context_str = "\n\n---\n\n".join(context_blocks)
            
            # 4. Build the final prompt string
            prompt_text = f"{INSTRUCTION}{context_str}\n\nQuestion: {question}\nAnswer:"
            
            case_prompts.append({
                "experiment": "1A",
                "query_id": query_id,
                "distractor_count": count,
                "prompt": prompt_text
            })
            
        # Save this specific case to its own isolated JSON file
        filename = f"{output_dir}/{count}_distractors.json"
        with open(filename, 'w') as file:
            json.dump(case_prompts, file, indent=4)
            
        print(f"Saved {len(case_prompts)} prompts to {filename}")

if __name__ == "__main__":
    # You can point this directly at your original eli5_good.json 
    # since it already contains the distractors!
    input_file = 'data/processed/eli5_good_with_noise.json' 
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}.")
    else:
        dataset = load_data(input_file)
        print("Generating Experiment 1A prompts (using Distractors)...")
        generate_experiment_1a_distractors(dataset)
        print("\nAll cases generated successfully in the 'prompts/exp1a_distractors/' directory.")