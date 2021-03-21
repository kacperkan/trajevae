import easydict

CONFIG = easydict.EasyDict()

CONFIG.seed = 1337

CONFIG.load_pretrained = True
CONFIG.learnable_initial_pose = False

CONFIG.num_training_steps = 240_000
CONFIG.batch_size = 64

CONFIG.t_his = 1

CONFIG.save_model_frequency = 500
CONFIG.log_img_frequency = 500
CONFIG.validation_frequency = 500
CONFIG.visualize_every_nth_pose = 20

CONFIG.lr = 0.0001
CONFIG.beta1 = 0.9
CONFIG.beta2 = 0.99
CONFIG.scheduler_step = 80_000
CONFIG.scheduler_gamma = 0.25

CONFIG.multimodal_threshold = 0.5
CONFIG.num_seeds = 1
CONFIG.actions = "all"
CONFIG.results_dir = "results/"
CONFIG.nk = 50  # num of different samples per pose
CONFIG.n_visualizations = 25
CONFIG.n_samples_per_visualization = 5

CONFIG.criterions = {
    "rec": {"class": "MSE", "params": {}},
    "kld": {"class": "OneDimensionalKLD", "params": {"beta": 0.1}},
}

CONFIG.experiment_subfolder = ""
CONFIG.target_seq_len = None
CONFIG.pseudo_past_frames = 25

CONFIG.standardize_data = False
CONFIG.deterministic = False
