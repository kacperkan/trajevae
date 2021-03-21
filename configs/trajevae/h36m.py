from ._base import CONFIG

CONFIG.experiment_name = "trajevae-h36m"
CONFIG.dataset_type = "human36m"

CONFIG.t_pred = 100

CONFIG.dct_components = 100
CONFIG.joint_dropout = 0.85
CONFIG.num_joints = 17
CONFIG.encoder_dropout = 0.1
CONFIG.decoder_dropout = 0.1

CONFIG.criterions["kld"]["params"]["beta"] = 0.01

CONFIG.joint_indices_to_use = [
    # 3,  # right ankle,
    # 6,  # left ankle,
    # 13,  # left wrist,
    # 16,  # right wrist
]
