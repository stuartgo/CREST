"""Best hyperparameters from Optuna sweeps, keyed by (model_type, location, temperature, single_day)."""

PARAMS = {
    # ── back / movement / multi-day ────────────────────────────────────────
    ("lstm",        "back", False, False): {"bidirectional": True,  "d_model": 473, "dropout": 0.4153, "learning_rate": 1.321e-4, "num_layers": 1},
    ("lstm",        "back", False, True):  {"bidirectional": False, "d_model": 335, "dropout": 0.3318, "learning_rate": 6.625e-4, "num_layers": 2},
    ("lstm",        "back", True,  False): {"bidirectional": False, "d_model": 296, "dropout": 0.0673, "learning_rate": 2.911e-4, "num_layers": 3},
    ("lstm",        "back", True,  True):  {"bidirectional": True,  "d_model": 230, "dropout": 0.3218, "learning_rate": 1.463e-3, "num_layers": 1},

    ("mamba",       "back", False, False): {"d_conv": 4, "d_model": 512, "d_state": 16, "expand": 4, "learning_rate": 4.905e-4},
    ("mamba",       "back", False, True):  {"d_conv": 3, "d_model":  64, "d_state": 64, "expand": 2, "learning_rate": 1.422e-4},
    ("mamba",       "back", True,  False): {"d_conv": 3, "d_model":  64, "d_state": 32, "expand": 4, "learning_rate": 3.895e-4},
    ("mamba",       "back", True,  True):  {"d_conv": 2, "d_model": 128, "d_state": 32, "expand": 2, "learning_rate": 4.656e-4},

    ("tcn",         "back", False, False): {"activation": "relu", "dropout": 0.1619, "kernel_size": 8, "learning_rate": 1.469e-4, "num_channels": [256, 256, 256]},
    ("tcn",         "back", False, True):  {"activation": "relu", "dropout": 0.0624, "kernel_size": 8, "learning_rate": 5.098e-4, "num_channels": [ 64,  64,  64]},
    ("tcn",         "back", True,  False): {"activation": "relu", "dropout": 0.0663, "kernel_size": 5, "learning_rate": 1.248e-3, "num_channels": [256, 256, 256]},
    ("tcn",         "back", True,  True):  {"activation": "relu", "dropout": 0.4424, "kernel_size": 4, "learning_rate": 1.893e-4, "num_channels": [512, 512, 512]},

    ("transformer", "back", False, False): {"d_model":  64, "dim_feedforward": 184, "dropout": 0.4416, "learning_rate": 2.402e-4, "num_layers": 2, "num_heads": 8},
    ("transformer", "back", False, True):  {"d_model": 128, "dim_feedforward": 225, "dropout": 0.1231, "learning_rate": 3.310e-4, "num_layers": 3, "num_heads": 4},
    ("transformer", "back", True,  False): {"d_model": 512, "dim_feedforward": 204, "dropout": 0.1329, "learning_rate": 2.467e-5, "num_layers": 4, "num_heads": 2},
    ("transformer", "back", True,  True):  {"d_model":  64, "dim_feedforward": 218, "dropout": 0.1569, "learning_rate": 1.247e-5, "num_layers": 6, "num_heads": 4},

    # ── thigh / movement / multi-day ───────────────────────────────────────
    ("lstm",        "thigh", False, False): {"bidirectional": True,  "d_model": 178, "dropout": 0.3848, "learning_rate": 1.894e-4, "num_layers": 3},
    ("lstm",        "thigh", False, True):  {"bidirectional": True,  "d_model": 344, "dropout": 0.2700, "learning_rate": 1.981e-4, "num_layers": 4},
    ("lstm",        "thigh", True,  False): {"bidirectional": True,  "d_model":  99, "dropout": 0.2125, "learning_rate": 1.918e-4, "num_layers": 1},
    ("lstm",        "thigh", True,  True):  {"bidirectional": False, "d_model": 372, "dropout": 0.0672, "learning_rate": 1.429e-4, "num_layers": 2},

    ("mamba",       "thigh", False, False): {"d_conv": 3, "d_model": 512, "d_state": 32, "expand": 2, "learning_rate": 2.035e-3},
    ("mamba",       "thigh", False, True):  {"d_conv": 3, "d_model":  64, "d_state": 32, "expand": 2, "learning_rate": 8.196e-5},
    ("mamba",       "thigh", True,  False): {"d_conv": 3, "d_model": 256, "d_state": 64, "expand": 4, "learning_rate": 3.067e-3},
    ("mamba",       "thigh", True,  True):  {"d_conv": 2, "d_model": 512, "d_state": 32, "expand": 2, "learning_rate": 1.141e-4},

    ("tcn",         "thigh", False, False): {"activation": "relu", "dropout": 0.2656, "kernel_size": 7, "learning_rate": 4.716e-4, "num_channels": [128, 128, 128]},
    ("tcn",         "thigh", False, True):  {"activation": "tanh", "dropout": 0.0706, "kernel_size": 8, "learning_rate": 1.298e-4, "num_channels": [256, 256, 256]},
    ("tcn",         "thigh", True,  False): {"activation": "relu", "dropout": 0.2578, "kernel_size": 7, "learning_rate": 8.549e-4, "num_channels": [ 64,  64,  64]},
    ("tcn",         "thigh", True,  True):  {"activation": "relu", "dropout": 0.2152, "kernel_size": 6, "learning_rate": 2.062e-4, "num_channels": [ 64,  64,  64]},

    ("transformer", "thigh", False, False): {"d_model":  64, "dim_feedforward": 254, "dropout": 0.1103, "learning_rate": 4.098e-4, "num_layers": 6, "num_heads": 4},
    ("transformer", "thigh", False, True):  {"d_model":  64, "dim_feedforward": 183, "dropout": 0.1001, "learning_rate": 1.000e-4, "num_layers": 3, "num_heads": 8},
    ("transformer", "thigh", True,  False): {"d_model":  64, "dim_feedforward": 237, "dropout": 0.2912, "learning_rate": 2.275e-4, "num_layers": 5, "num_heads": 4},
    ("transformer", "thigh", True,  True):  {"d_model":  64, "dim_feedforward": 139, "dropout": 0.2353, "learning_rate": 7.223e-5, "num_layers": 2, "num_heads": 8},
}


def get_params(model_type, location, temperature, single_day):
    key = (model_type, location, temperature, single_day)
    if key not in PARAMS:
        raise ValueError(f"No hyperparameters found for key: {key}")
    params = PARAMS[key].copy()
    # TCN's d_model is derived from the channel width
    if model_type == "tcn":
        params["d_model"] = params["num_channels"][-1]
    return params
