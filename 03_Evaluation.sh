#!/bin/bash

model="dmis-lab/biobert-base-cased-v1.2"

train_file="./results/ensemble.iob.txt"
test_file="./sample_files/test.tsv"
output_dir="./results/augmented_result"
label_list="B-Disease,I-Disease,B-Chemical,I-Chemical,B-Species,I-Species,B-Gene,I-Gene,B-CellLine,I-CellLine,B-Variant,I-Variant,O"

loss_func="sce"
epochs=40
min_epochs=20
lr=3e-5

cmd="python ner.py
     --model_name_or_path ${model}
     --do_train --train_file ${train_file}
     --do_predict --predict_file ${test_file}
     --output_dir ${output_dir}/${name}_result
     --label_list ${label_list}
     --loss_func ${loss_func}
     --min_epochs ${min_epochs}
     --epochs ${epochs}
     --lr ${lr}"

echo $cmd
eval $cmd
