from ._base import CONFIG

CONFIG.experiment_name = "moglow-h36m"
CONFIG.dataset_type = "human36m"


CONFIG.num_joints = 17
CONFIG.limb_indices = []
CONFIG.joint_dropout = 0.85
CONFIG.t_pred = 110
