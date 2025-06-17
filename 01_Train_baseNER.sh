#!/bin/bash

plms=(
  "biobert dmis-lab/biobert-base-cased-v1.2"
  "biomedbert microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
  "bioformer bioformers/bioformer-16L"
)
train_file="./sample_files/small_data.tsv"
predict_file="./sample_files/augmented_data.tsv"
output_dir="./results"
label_list="B-Disease,I-Disease,B-Chemical,I-Chemical,B-Species,I-Species,B-Gene,I-Gene,B-CellLine,I-CellLine,B-Variant,I-Variant,O"

min_epochs=20
epochs=50
lr=3e-5
batch_size=2
seed=45

output_list=()

mkdir -p "$output_dir"

for plm in "${plms[@]}"
do
  read name model <<< "$plm"
  cmd="python ner.py
       --model_name_or_path ${model}
       --do_train --train_file ${train_file}
       --do_predict --predict_file ${predict_file}
       --output_dir ${output_dir}/${name}_result
       --label_list ${label_list}
       --min_epochs ${min_epochs}
       --epochs ${epochs}
       --batch_size ${batch_size}
       --lr ${lr}
       --max_seq_length 512
       --seed ${seed}
       --early_stopping_threshold 0.0001"
  echo $cmd
  eval $cmd
  output_list+=(${output_dir}/${name}_result/predictions.txt)
done

mkdir -p "${output_dir}"
printf "%s\n" "${output_list[@]}" > "${output_dir}/base_ner.list.txt"
