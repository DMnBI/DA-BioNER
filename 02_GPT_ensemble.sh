#!/bin/bash

output_dir="./results"
list_file="${output_dir}/base_ner.list.txt"

# === Required: Set your GPT and UMLS api keys here ===
gpt_api_key="{add_gpt_key}"
umls_api_key="{add_umls_key}"

cmd="python gpt_ensemble.py --list_file $list_file --output_dir $output_dir --api_key $gpt_api_key --umls_api_key $umls_api_key --overwrite_preprocessed_file --overwrite_gpt_output"
echo $cmd
eval $cmd
