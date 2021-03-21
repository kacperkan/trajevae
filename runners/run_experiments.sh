#!/bin/bash


run_single() {
    folder_name=$1

    if [ ! -f "outputs/models/${folder_name}/model.pt" ]
    then
        echo "Experiment ${folder_name} doesn't exist!"
        return
    fi

    dataset=$2
    config_name=$3
    module_name=$4
    python -m \
        trajevae.evaluators.experiment \
        data/processed/${dataset} \
        --config_name ${config_name} \
        --module_name ${module_name} \
        --without_visualization
}

echo "Running RNN ..."
run_single rnn-baseline-h36m human36m-3d rnn.h36m baselines.rnn.Model

echo "Running VAE ..."
run_single vae-baseline-h36m human36m-3d vae.h36m baselines.vae.Model

echo "Running CVAE ..."
run_single cvae-baseline-h36m human36m-3d cvae.h36m baselines.cvae.Model
