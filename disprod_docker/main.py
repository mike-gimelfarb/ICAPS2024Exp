import json
import sys
import signal
import time
import traceback

# sys.path.append('/home/test/pyRDDLGym')
import numpy as np
from pyRDDLGym.Core.Env import RDDLEnv
from pyRDDLGym.Core.Policies.Agents import NoOpAgent
from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

import copy
from functools import partial
from gym.spaces import Dict, Box

# for JAX backend:
# from pyRDDLGym.Core.Jax.JaxRDDLSimulator import JaxRDDLSimulator

############################################################
# IMPORT THE AGENT AND OTHER DEPENDENCIES OF YOUR SOLUTION #
from utils import helpers, heuristics
from functools import partial
from planners.disprod import ContinuousDisprod
import jax
from utils.common_utils import prepare_config
import os
import multiprocessing as mp
from concurrent.futures import as_completed, ProcessPoolExecutor      

DISPROD_NOISE_VARS = ["disprod_eps_norm", "disprod_eps_uni"]
############################################################

HOUR = 3600
def signal_handler(signum, frame):
    raise Exception("Timed out!")


# MAIN INTERACTION LOOP #
def main(env, inst, method_name=None, episodes=1, time_ps=1):
    print(f'preparing to launch instance {inst} of domain {env}...')

    # get the environment info
    EnvInfo = RDDLRepoManager().get_problem(env)

    # set up the environment class, choose instance 0 because every example has at least one example instance
    log = False if method_name is None else True
    current_dir = os.path.dirname(os.path.abspath(__file__))
    myEnv = RDDLEnv.RDDLEnv(domain=EnvInfo.get_domain(),
                            instance=EnvInfo.get_instance(inst),
                            enforce_action_constraints=False,
                            debug=True,
                            log_path=os.path.join(current_dir, 'outputs', f'{env}_{inst}_disprod_True_{time_ps}.log'))
                            # backend=JaxRDDLSimulator)

    # initialization of the method init timer
    budget = myEnv.horizon * time_ps # myEnv.Budget
    init_budget = 2 * HOUR * time_ps
    init_timed_out = False
    signal.signal(signal.SIGALRM, signal_handler)

    # default noop agent, do not change
    defaultAgent = NoOpAgent(action_space=myEnv.action_space,
                        num_actions=myEnv.max_allowed_actions)

    signal.setitimer(signal.ITIMER_REAL, init_budget)
    start = time.time()
    try:
        ################################################################
        # Initialize your agent here:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_name = "_".join(env.lower().split('_')[:-1])
        cfg = prepare_config(env_name, f"{current_dir}/config")

        checkpoint = time.time()
        print(f"[Time left: {init_budget - (checkpoint-start)}] Loading config {env_name} from path {current_dir}/config")

        # Don't reparameterize the RDDL expressions if planner uses sampling
        domain_path = EnvInfo.get_domain()
        instance_path = EnvInfo.get_instance(inst)
        
        # Setup rddl_model for sampling mode, and reparam_rddl_model for NV and complete mode
        rddl_model = helpers.gen_model(domain_path, instance_path, False)
        reparam_rddl_model = helpers.gen_model(domain_path, instance_path, True)

        # For large RecSim instances.       
        if len(myEnv.action_space) > 250 or len(myEnv.observation_space) > 2500:
            fallback = True
            cfg["no_var"]["n_restarts"] = 2
            cfg["sampling"]["n_restarts"] = 2
            cfg["depth"] = 2
            cfg["no_var"]["max_grad_steps"]=2
            cfg["sampling"]["max_grad_steps"]=2
        else:
            fallback = False

        heuristic_scan = False if fallback or cfg["skip_search"] else True
        
        
        if fallback and cfg["env_name"] == "recsim":
            g_obs_keys, ac_dict_fn, cfg_env = helpers.prepare_cfg_env_fallback_recsim(myEnv, cfg, rddl_model)
            del reparam_rddl_model
        else:
        
            # Setup cfg_env for sampling mode and reparam_cfg_env for NV and complete mode
            # TODO: Check if reparam_obs and reparam_a keys are different than normal?
            g_obs_keys, ga_keys, ac_dict_fn, cfg_env = helpers.prepare_cfg_env(env, myEnv, rddl_model, cfg)
            _, _, _, reparam_cfg_env = helpers.prepare_cfg_env(env, myEnv, reparam_rddl_model, cfg)
             
        # Get a dummy obs
        dummy_state = myEnv.reset()
        dummy_obs = np.array([dummy_state[i] for i in g_obs_keys])

        # Instance 1 of marsrover works better when prewarmed using all zeros.
        if cfg["env_name"] == "marsrover" and inst == "1c":
            dummy_obs = np.zeros_like(dummy_obs)

        # Setup default agent depending on the default mode
        agent_setup_start = time.time()
        if cfg["mode"] == "sampling":
            agent = ContinuousDisprod(cfg, rddl_model, cfg_env)
            lrs_to_scan = agent.pre_warm(dummy_obs)
        else:
            agent = ContinuousDisprod(cfg, reparam_rddl_model, reparam_cfg_env)
            lrs_to_scan = agent.pre_warm(dummy_obs)
            
        agent_key = jax.random.PRNGKey(cfg["seed"])
        prev_ac_seq, agent_key = agent.reset(agent_key)
        if cfg[cfg["mode"]]["overwrite_lrs"]:
            lrs_to_scan = cfg[cfg["mode"]]["lrs_to_scan"]
        
        agent_setup_end = time.time()

        time_required_for_agent_setup = agent_setup_end-agent_setup_start
        print(f"[Time left: {init_budget - (agent_setup_end - start)}] Basic agent initialized. Time taken: {time_required_for_agent_setup}")

        checkpoint = time.time()
        # See if we have enough time for performing a scan
        time_for_scan = init_budget - (checkpoint - start) - (time_required_for_agent_setup) - (budget + 60)
        if time_for_scan < budget:
            heuristic_scan = False
        # Perform heuristic scans       
        ##################################################################
        # H1: Search across different modes and see which is better
        ##################################################################
        if heuristic_scan:
            combs = []
            for mode in ["no_var", "sampling"]:
                for s_wt in [3, 5]:
                    #for restart in [cfg[mode]["n_restarts"], cfg[mode]["n_restarts"]/2]:
                    restart = cfg[mode]["n_restarts"]
                    if mode == "no_var":
                        model = reparam_rddl_model
                        _cfg_env = reparam_cfg_env
                    else:
                        model = rddl_model
                        _cfg_env = cfg_env
                    combs.append((mode, model, _cfg_env, s_wt, restart))
            scan_res = []
            
            heuristic_fn = partial(heuristics.compute_score_stats, domain_path, instance_path, g_obs_keys, ga_keys, ac_dict_fn, n_episodes=5, time_budget=time_for_scan, time_ps=time_ps)

            # JAX doesn't fork with fork context which is default for Linux. Start a spawn context explicitly.
            # context = mp.get_context("spawn")
            # with ProcessPoolExecutor(mp_context=context) as executor:
            #     jobs = [executor.submit(heuristic_fn, copy.deepcopy(cfg), mode, model, _cfg_env, s_wt, restart) for (mode, model, _cfg_env, s_wt, restart) in combs]

            #     for job in as_completed(jobs):
            #         result = job.result()
            #         scan_res.append(result)

            for  (mode, model, _cfg_env, s_wt, restart) in combs:
                scan_res.append(heuristic_fn(copy.deepcopy(cfg), mode, model, _cfg_env, s_wt, restart))

            scan_res = sorted(scan_res, key=lambda x: x[0])
            
            better_mode, better_weight, better_restart = scan_res[-1][1], scan_res[-1][2], scan_res[-1][3] 
            checkpoint = time.time()
            print(f"[Time left: {init_budget - (checkpoint - start)}] Heuristic scan complete.")

            if better_mode != cfg["mode"] or better_weight != cfg["logic_kwargs"]["weight"] or better_restart != cfg[better_mode]["n_restarts"]:
                cfg["mode"] = better_mode
                cfg["logic_kwargs"]["weight"] = better_weight
                cfg[better_mode]["n_restarts"] = better_restart
                if cfg["mode"] == "sampling":
                    new_agent = ContinuousDisprod(cfg, rddl_model, cfg_env)
                    new_lrs_to_scan = agent.pre_warm(dummy_obs)
                else:
                    new_agent = ContinuousDisprod(cfg, reparam_rddl_model, reparam_cfg_env)
                    new_lrs_to_scan = agent.pre_warm(dummy_obs)
                agent = new_agent
                if cfg[cfg["mode"]]["overwrite_lrs"]:
                    lrs_to_scan = cfg[cfg["mode"]]["lrs_to_scan"]
                else:
                    lrs_to_scan = new_lrs_to_scan
                checkpoint = time.time()
                print(f"[Time left: {init_budget - (checkpoint - start)}] Found better config during scan. New agent initialized.")
        
        
        ##############################################################
    except Exception:
        print(traceback.format_exc())
        finish = time.time()
        print('Initialization timed out', finish - start, ' seconds)')
        # print('This domain will continue exclusively with default actions!')
        init_timed_out = True

    signal.signal(signal.SIGALRM, signal_handler)
    
    history = []
    for episode in range(episodes):
        total_reward = 0
        state = myEnv.reset()
        # timed_out = False if init_timed_out==False else True
        timed_out = False
        elapsed = budget
        start = 0
        for step in range(myEnv.horizon):

            # action selection:
            if not timed_out:
                signal.setitimer(signal.ITIMER_REAL, elapsed)
                start = time.time()
                try:
                    #################################################################
                    # replace the following line of code with your agent call
                    obs_array = np.array([state[i] for i in g_obs_keys])
                    ac_array, k_idx, prev_ac_seq, agent_key, _ = agent.choose_action(obs_array, prev_ac_seq, agent_key, lrs_to_scan)
                    action = ac_dict_fn(ac_array, k_idx)

                    #################################################################
                    finish = time.time()
                except Exception:
                    print(traceback.format_exc())
                    finish = time.time()
                    print('Timed out! (', finish-start, ' seconds)')
                    print('This episode will continue with default actions!')
                    action = defaultAgent.sample_action()
                    timed_out = True
                    elapsed = 0
                    if not timed_out:
                        elapsed = elapsed - (finish-start)
            else:
                action = defaultAgent.sample_action()

            next_state, reward, done, info = myEnv.step(action)
            total_reward += reward

            print()
            print(f'step       = {step}')
            print(f'state      = {state}')
            print(f'action     = {action}')
            print(f'next state = {next_state}')
            print(f'reward     = {reward}')

            state = next_state

            if done:
                break

        print(f'episode {episode+1} ended with reward {total_reward} after {budget-elapsed} seconds')
        history.append(total_reward)
    myEnv.close()

    result =  {
            'mean': np.mean(history),
            'median': np.median(history),
            'min': np.min(history),
            'max': np.max(history),
            'std': np.std(history)
    }
    
    # dump all history to files
    print('writing logs...')
    with open(os.path.join(current_dir, 'outputs', f'{env}_{inst}_disprod_True_{time_ps}.json'), 'w') as fp:
        json.dump(result, fp, indent=4)
    print('writing logs complete!')


    ########################################
    # CLEAN UP ANY RESOURCES YOU HAVE USED #


    ########################################


# Command line interface, DO NOT CHANGE
if __name__ == "__main__":
    args = sys.argv
    print(args)
    
    env, inst, time_ps = args[1:4]
    
    episodes = 20
    method_name = 'disprod'
    try:
        time_ps = int(time_ps)
    except:
        raise ValueError("time must be an integer value argument, received: " + time_ps)
    
    main(env, inst, method_name, episodes, time_ps)

