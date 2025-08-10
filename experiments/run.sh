#!/bin/bash
datasets=("adult" "law")
policies=("full_data" "wo_proattr" "fair")
algorithms=("one" "two")
models=("GNB" "LR")
commands=()
seeds=(1029 42 13 729 333)
for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
        for algo in "${algorithms[@]}"; do
            for dataset in "${datasets[@]}"; do
                for policy in "${policies[@]}"; do
                    commands+=("python3 partial_feedback.py --dataset $dataset --policy $policy --algorithm $algo --model $model --seed $seed")
                done
            done
        done
    done
done
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    $cmd &  
done

datasets=("adult" "law")
policies=("random" "protected")
algorithms=("one")
models=("LR")
commands=()
seeds=(1029 42 13 729 333)
for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
        for algo in "${algorithms[@]}"; do
            for dataset in "${datasets[@]}"; do
                for policy in "${policies[@]}"; do
                    commands+=("python3 partial_feedback.py --dataset $dataset --policy $policy --algorithm $algo --model $model --seed $seed")
                done
            done
        done
    done
done
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    $cmd &  
done