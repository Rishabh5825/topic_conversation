import json 
import os
import sys
import re
import openai
from utils.benchmark_utils import *
import pandas as pd
import numpy as np
import tqdm
import io
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

input_folder = 'data/example_aug'
output_folder = "tmp/prompts"
prep_folder(output_folder)
meeting_files = [x for x in locate_suffix_files(input_folder, suffix="json")]
print(f"Found {len(meeting_files)} meeting files.")

prompts = generate_relevance_prompts(meeting_files)
prompts = prompts[:30]
prompt_path = write_prompts_to_json(prompts, output_folder)
print(f"Prompts written to {prompt_path}")

try:
    with open(prompt_path, 'r') as f:
        prompt_entries = [json.loads(line) for line in f]
    print(f"Loaded {len(prompt_entries)} prompts.")
except FileNotFoundError:
    print(f"Error: Prompt file not found at {prompt_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {prompt_path}")
    sys.exit(1)


prompt_list = [d["prompt"] for d in prompt_entries]

YOUR_API_KEY = os.getenv("PERPLEXITY_API_KEY")
client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

def run_prompts_perplexity(prompt_entries, model="Gemini 2.5 Pro"):
    results = {}
    for idx, entry in enumerate(prompt_entries):
        prompt_id = entry.get('promptId') or entry.get('prompt_id') or str(idx)
        prompt_text = entry.get("prompt") or entry.get("text") or ""

        if not prompt_text:
            print(f"Skipping empty prompt {prompt_id}")
            continue

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=2048,
            temperature=0.0
        )
        reply = response.choices[0].message.content
        print(f"Processed prompt {idx+1}/{len(prompt_entries)} (ID: {prompt_id})")
        results[prompt_id] = reply

    result_path = os.path.join(output_folder, "perplexity_results.jsonl")
    with open(result_path, 'w') as fout:
        for pid, resp in results.items():
            fout.write(json.dumps({"promptId": pid, "response": resp}) + "\n")

    print(f"Results saved to {result_path}")
    return result_path

result_path = run_prompts_perplexity(prompt_entries, model="sonar-pro")

# def prompt_styles():
#     # We treat whole summary text as the response part
#     response_parts = {
#         'summary_text': 'text',
#     }
#     list_to_remove = ['']
#     return response_parts, list_to_remove

# def parse_responses(all_responses, df_gt, list_to_remove, response_parts):
#     parsed = []
#     error_entries = []
#     for res in all_responses:
#         prompt_id = res['promptId']
#         response_text = res['response'].strip()

#         try:
#             # Heuristic: detect relevant or not based on certain cues in text
#             # Here we just mark "Relevant" if keywords appear, else "Not Discussed"
#             keywords = ['reference', 'mentions', 'relevant', 'discussed', 'noted', 'described']
#             label = "Not Discussed"
#             if any(word in response_text.lower() for word in keywords):
#                 label = "Relevant"

#             parsed.append({
#                 'promptId': prompt_id,
#                 'predicted_label': label,
#                 'raw_response': response_text
#             })

#         except Exception as e:
#             error_entries.append((prompt_id, f"Parsing error: {str(e)}"))

#     df_parsed = pd.DataFrame(parsed)
#     return df_parsed, error_entries

# # The rest of your existing workflow:
# # - Load ground truth df_gt using get_gt_data()
# # - Join with parsed results via join_results()
# # - Calculate metrics with calculate_metrics()
# # - Display them with display_metrics()

# # Usage after running LLM inference and saving results to `result_path`
# response_parts, list_to_remove = prompt_styles()
# all_responses = get_raw_completions(result_path)
# df_model, errors = parse_responses(all_responses, df_gt, list_to_remove, response_parts)

# if errors:
#     for err in errors:
#         print(f"Error in prompt {err[0]}: {err[1]}")
#     raise ValueError("Resolve above parsing errors before continuing")

# df_all_join, list_buckets, threshold_pairs = join_results(df_gt, df_model, 'your_output_name', output_folder)
# df_all_join['prompt_size'] = df_all_join['promptId'].str.split('_').str[-2]

# for psg, psdf in df_all_join.groupby('prompt_size'):
#     for b in list_buckets:
#         for p in threshold_pairs[b]:
#             title = '\n{} Evaluation Results for {} snippets; classification {}; threshold {}\n{}'.format(
#                 '='*10, psg, b, p, '='*10)
#             metrics, metrics_binary = calculate_metrics(psdf, b, p)
#             display_metrics(metrics, metrics_binary, title=title)


# Path to your JSON file containing prompts
# json_file_path = 'tmp/prompts/prompts_tcr_10.json'
# with open(json_file_path, 'r', encoding='utf-8') as f:
#     lines = f.readlines()
    
# for i, line in enumerate(lines):
#     try:
#         item = json.loads(line)
#         print(item)
#         prompt = item.get('prompt', '')
#         char_count = len(prompt)
#         approx_tokens = char_count // 4
#         print(f"Prompt {i+1}: Characters = {char_count}, Approx Tokens = {approx_tokens}")
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON on line {i+1}: {e}")
