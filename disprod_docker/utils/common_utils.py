import numpy as onp
from matplotlib import pyplot as plt
from matplotlib import animation
import importlib
import random
import jax.numpy as jnp
from jax import random as jax_random
from omegaconf import OmegaConf
import os
from datetime import date
import time
from pathlib import Path

# Argmax with random tie-breaks
# The 0th-restart is always set to the previous solution
def random_argmax(key, x, pref_idx=0):
    try:
        options = jnp.where(x == jnp.nanmax(x))[0]
        val = 0 if 0 in options else jax_random.choice(key, options)
    except:
        val = jax_random.choice(key, jnp.arange(len(x)))
        print(f"All restarts where NaNs. Randomly choosing {val}.")
    finally:
        return val


# Miscellaneous functions
# Dynamically load a function from a module
def load_method(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

# Print to file or stdout
def print_(x, path=None, flush=True):
    if path is None:
        print(x)
    else:
        print(x, file=open(path, 'a'), flush=flush)


# Seed the libraries
def set_global_seeds(seed, env=None):
    onp.random.seed(seed)
    random.seed(seed)

    if env:
        env.seed(seed)
        env.action_space.seed(seed)


def load_config_if_exists(path, log_path):
    if os.path.exists(path):
        print_(f"Using config file : {path}", log_path)
        return OmegaConf.load(path)
    print_(f"Requested config file not found at : {path}. Skipping", log_path)
    return {}

def prepare_config(env_name, cfg_path=None, log_path=None):
    config_files = [f"{cfg_path}/default.yaml",
                    f"{cfg_path}/{env_name}.yaml"
                    ]
    config = {}
    for config_file in config_files:
        config = OmegaConf.merge(config, load_config_if_exists(config_file, log_path))
    return OmegaConf.to_container(config)

def update_config_with_args(cfg, args, base_path):
    # Non-boolean keys
    keys_to_update = ["seed", "log_file", "depth", "n_episodes", "alg", "inst"]
    
    for key in keys_to_update:
        if args.__contains__(key) and getattr(args,key) is not None:
            cfg[key] = getattr(args, key)
     
    # Update n_restarts in our planner
    if args.__contains__("n_restarts") and getattr(args,"n_restarts") is not None:
        cfg["disprod"]["n_restarts"] = getattr(args, "n_restarts")
        
    # First parse mode
    mode = cfg["mode"]
    
    # Update n_restarts in our planner
    if args.__contains__("n_restarts") and getattr(args,"n_restarts") is not None:
        cfg[mode]["n_restarts"] = getattr(args, "n_restarts")
    
    # Update step-size in our planner        
    if args.__contains__("step_size") and getattr(args,"step_size") is not None:
        cfg[mode]["step_size"] = getattr(args, "step_size")
        
    if args.__contains__("step_size_var") and getattr(args,"step_size_var") is not None:
        cfg[mode]["step_size_var"] = getattr(args, "step_size_var")
        
    if args.__contains__("sop") and getattr(args,"sop") is not None:
        cfg[mode]["sop"] = getattr(args, "sop")

    if args.__contains__("opt") and getattr(args,"opt") is not None:
        cfg[mode]["optimizer"] = getattr(args, "opt")

    # Boolean keys
    boolean_keys = ["render", "headless", "save_as_gif"]
    for key in boolean_keys:
        if args.__contains__(key) and getattr(args,key) is not None:
            cfg[key] = getattr(args, key).lower() == "true"

    # If run_name is set, the update in config. Else set default value to {running_mode}_{current_time}
    if args.__contains__("run_name") and getattr(args, "run_name"):
        cfg["run_name"] = args.run_name
    else:
        today = int(time.time())
        run_name = f"{today}-{mode}"
        cfg["run_name"] = run_name

    return cfg

def setup_output_dirs(cfg, run_name, base_path):   
    base_dir = f"{base_path}/results/{cfg['env_name']}/planning/{run_name}"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    cfg["results_dir"] = base_dir

    log_dir = f"{base_dir}/logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cfg["log_dir"] = log_dir
    cfg["log_file"] = f"{log_dir}/debug.log"

