import easydict

CONFIG = easydict.EasyDict()

CONFIG.seed = 1337

CONFIG.load_pretrained = True
CONFIG.learnable_initial_pose = False

CONFIG.t_his = 1

CONFIG.save_model_frequency = 3000
CONFIG.log_img_frequency = 3000
CONFIG.validation_frequency = 3000
CONFIG.visualize_every_nth_pose = 20

CONFIG.multimodal_threshold = 0.5
CONFIG.num_seeds = 1
CONFIG.actions = "all"
CONFIG.results_dir = "results/"
CONFIG.nk = 50  # num of different samples per pose
CONFIG.n_visualizations = 25
CONFIG.n_samples_per_visualization = 5

CONFIG.criterions = {}

CONFIG.experiment_subfolder = ""

CONFIG.hidden_channels = 512
CONFIG.K = 16
CONFIG.actnorm_scale = 1.0
CONFIG.flow_permutation = "invconv"
CONFIG.flow_coupling = "affine"
CONFIG.network_model = "LSTM"
CONFIG.num_layers = 2
CONFIG.LU_decomposed = True
CONFIG.distribution = "normal"
CONFIG.standardize_data = False


CONFIG.seqlen = 10
CONFIG.n_lookahead = 0
CONFIG.frame_dropout = 0.7

CONFIG.lr = 0.0001
CONFIG.beta1 = 0.9
CONFIG.beta2 = 0.999
CONFIG.scheduler_step = 80_000
CONFIG.scheduler_gamma = 0.25

CONFIG.num_training_steps = 80_000
CONFIG.max_grad_clip = 5
CONFIG.max_grad_norm = 100
CONFIG.batch_size = 64

CONFIG.experiment_subfolder = ""
CONFIG.target_seq_len = None
CONFIG.pseudo_past_frames = 25

CONFIG.validation_batch_size = 256

CONFIG.experiment_subfolder = ""
CONFIG.target_seq_len = None
CONFIG.pseudo_past_frames = 25

CONFIG.standardize_data = False
CONFIG.deterministic = False
