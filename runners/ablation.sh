#!/bin/sh

# base
python -m trajevae.trainers.base_trainer \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --disable_masking_future_poses \
    --disable_dct \
    --disable_learnable_prior \
    --disable_data_augmentation \
    --debug \
    --suffix basic 

python -m trajevae.evaluators.experiment \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --disable_masking_future_poses \
    --disable_dct \
    --disable_learnable_prior \
    --disable_data_augmentation \
    --debug \
    --suffix basic \
    --without_visualization


# dct
python -m trajevae.trainers.base_trainer \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --disable_masking_future_poses \
    --disable_learnable_prior \
    --disable_data_augmentation \
    --debug \
    --suffix dct 

python -m trajevae.evaluators.experiment \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --disable_masking_future_poses \
    --disable_learnable_prior \
    --disable_data_augmentation \
    --debug \
    --suffix dct  \
    --without_visualization


# dct + prior
python -m trajevae.trainers.base_trainer \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --disable_masking_future_poses \
    --disable_data_augmentation \
    --debug \
    --suffix dct_prior

python -m trajevae.evaluators.experiment \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --disable_masking_future_poses \
    --disable_data_augmentation \
    --debug \
    --suffix dct_prior \
    --without_visualization


# dct + prior + augmentation
python -m trajevae.trainers.base_trainer \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --disable_masking_future_poses \
    --debug \
    --suffix dct_prior_aug

python -m trajevae.evaluators.experiment \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --disable_masking_future_poses \
    --debug \
    --suffix dct_prior_aug \
    --without_visualization


# dct + prior + augmentation + masking
python -m trajevae.trainers.base_trainer \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --debug \
    --suffix dct_prior_aug_mask

python -m trajevae.evaluators.experiment \
    data/processed/human36m-3d/ \
    --config_name trajevae.h36m \
    --module_name pose2pose.Pose2Pose  \
    --debug \
    --suffix dct_prior_aug_mask \
    --without_visualization



