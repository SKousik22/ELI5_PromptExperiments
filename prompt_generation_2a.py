import json
import os

# The exact instruction prompt for your LLM
INSTRUCTION = "Explain the answer to the following question as if I were a 5-year-old. Base your explanation strictly on the provided context. Provide ONLY your final explanation. Do not generate any extra documents, follow-up questions, or separators."

def load_data(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as file:
        return json.load(file)

def generate_experiment_2a(dataset):
    """
    Experiment 2A: Impact of Gold Positioning
    Shifts the Gold document across specific depth intervals within the distractors.
    """
    # The distinct configurations: Total Distractors -> Insertion Positions
    configurations = {
        9: [0, 4, 9],               # 0%, 50%, 100% depth
        19: [0, 4, 9, 14, 19],      # 0%, 25%, 50%, 75%, 100% depth
        29: [0, 3, 7, 10, 14, 18, 22, 25, 29] # ~12.5% intervals
    }
    
    # Create the output directory
    output_dir = "prompts/exp2a"
    os.makedirs(output_dir, exist_ok=True)

    # Outer loops: Total context size and the specific position index
    for total_S, positions in configurations.items():
        for pos in positions:
            case_prompts = []
            depth_pct = (pos / total_S) * 100 if total_S > 0 else 0
            
            for entry in dataset:
                query_id = entry.get('query_id', 'unknown')
                question = entry.get('question', '')
                
                # 1. Safely Extract the Gold text
                gold_data = entry.get('gold', [])
                gold_text = " ".join([d.get("text", "") for d in gold_data if isinstance(d, dict) and "text" in d])
                
                # 2. Safely Extract the Distractors text
                distractor_data = entry.get('distractors', [])
                all_distractors = [d.get("text", "") for d in distractor_data if isinstance(d, dict) and "text" in d]
                
                # Slice the array to the required total context size
                S_docs = all_distractors[:total_S]
                
                if len(S_docs) < total_S:
                    print(f"Warning: Entry {query_id} only has {len(S_docs)} distractors, but {total_S} requested.")
                
                # 3. Inject Gold into the specific position
                # We use .copy() so we don't mutate the original array
                context_blocks = S_docs.copy()
                context_blocks.insert(pos, gold_text)
                
                # 4. Construct the layout
                context_str = "\n\n---\n\n".join(context_blocks)
                
                # Build the final prompt string
                prompt_text = f"{INSTRUCTION}{context_str}\n\nQuestion: {question}\nAnswer:"
                
                case_prompts.append({
                    "experiment": "2A",
                    "query_id": query_id,
                    "total_distractors": total_S,
                    "gold_position_index": pos,
                    "depth_percentage": round(depth_pct, 1),
                    "prompt": prompt_text
                })
                
            # Save this specific positioning case to its own isolated JSON file
            filename = f"{output_dir}/{total_S}_distractors_pos_{pos}.json"
            with open(filename, 'w') as file:
                json.dump(case_prompts, file, indent=4)
                
            print(f"Saved {len(case_prompts)} prompts to {filename}")

if __name__ == "__main__":
    # Pointing directly to your original dataset since it has the native distractors
    input_file = 'data/processed/eli5_good_with_noise.json' 
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}.")
    else:
        dataset = load_data(input_file)
        print("Generating Experiment 2A prompts (Positioning)...")
        generate_experiment_2a(dataset)
        print("\nAll 17 positioning cases generated successfully in the 'exp_2a/prompts' directory.")