import json
import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import _parse_config_file, _load_config
from pyRDDLGym.Core.Policies.Agents import NoOpAgent, RandomAgent

from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

from baselines.gurobiplan import gurobi_policy
from baselines.jaxplan import jax_policy

# load global config
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
_, global_args = _parse_config_file(os.path.join(ROOT_PATH, 'global.cfg'))


def main(domain, instance, method, online, do_tune):
    outputpath = os.path.join(ROOT_PATH, 'outputs', method, 
                              f'{domain}_{instance}_{method}_{online}')
    
    # create the environment
    EnvInfo = RDDLRepoManager().get_problem(domain)   
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(instance),
                  enforce_action_constraints=True,
                  log=True, 
                  log_path=outputpath)
    
    # load the config file with planner settings
    config, args = _parse_config_file(os.path.join(ROOT_PATH, 'configs', f'{domain}.cfg'))
    
    # override default config settings here
    args['train_seconds'] = global_args['train_seconds_per_epoch'] * (1 if online else env.horizon)
    if not online:
        args['rollout_horizon'] = None
    planner_args, plan_args, train_args = _load_config(config, args)
    
    # dispatch to policy creation method
    if method == 'jax':
        policy = jax_policy(env, online, do_tune, 
                            config, args, planner_args, plan_args, train_args, 
                            outputpath, global_args)
        ground_state = False
        
    elif method == 'gurobi':
        policy = gurobi_policy(env, online, do_tune, args, outputpath, global_args)
        ground_state = True
    
    elif method in ['noop', 'random']:
        if method == 'noop':
            policy = NoOpAgent(action_space=env.action_space,
                               num_actions=env.numConcurrentActions)
        else:
            policy = RandomAgent(action_space=env.action_space,
                                 num_actions=env.numConcurrentActions)
        ground_state = True
    
    else:
        raise Exception(f'Invalid method {method}.')
    
    # evaluation
    result = policy.evaluate(env, 
                             verbose=True, 
                             episodes=global_args['runs'], 
                             ground_state=ground_state)
    
    # dump all history to files
    with open(outputpath + '.json', 'w') as fp:
        json.dump(result, fp, indent=4)
    env.close()


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 5:
        domain, instance, method, online, do_tune = args[1:6]
    else:
        domain, instance, method, online, do_tune = 'CooperativeRecon_MDP_ippc2011', '3', 'gurobi', True, True
    online = online in {'True', 'true', True, '1', 1}
    do_tune = do_tune in {'True', 'true', True, '1', 1}
    
    main(domain, instance, method, online, do_tune)
    
