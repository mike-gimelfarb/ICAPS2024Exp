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
_, global_args = _parse_config_file(os.path.join(ROOT_PATH, 'baselines', 'global.cfg'))


def main(domain, instance, method, online, tuning, time):
    outputpath = os.path.join(ROOT_PATH, 'outputs', method, 
                              f'{domain}_{instance}_{method}_{online}_{time}')
    
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
    args['train_seconds'] = time * (1 if online else env.horizon)
    if not online:
        args['rollout_horizon'] = None
    planner_args, plan_args, train_args = _load_config(config, args)
    
    # dispatch to policy creation method
    if method == 'jaxplan':
        policy = jax_policy(env, online, tuning, 
                            config, args, planner_args, plan_args, train_args, 
                            outputpath, global_args)
        
    elif method == 'gurobiplan':
        policy = gurobi_policy(env, online, tuning, args, outputpath, global_args)
    
    elif method in ['noop', 'random']:
        if method == 'noop':
            policy = NoOpAgent(action_space=env.action_space,
                               num_actions=env.numConcurrentActions)
        else:
            policy = RandomAgent(action_space=env.action_space,
                                 num_actions=env.numConcurrentActions)
    
    else:
        raise Exception(f'Invalid method {method}.')
    
    # evaluation
    result = policy.evaluate(env, verbose=False, episodes=global_args['episodes'])
    
    # dump all history to files
    with open(outputpath + '.json', 'w') as fp:
        json.dump(result, fp, indent=4)
    env.close()


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 6:
        domain, instance, method, online, tuning, time = args[1:7]
    else:
        domain, instance, method, online, tuning, time = \
            'Wildfire_MDP_ippc2014', '1', 'jaxplan', False, False, 1
    online = online in {'True', 'true', True, '1', 1}
    tuning = tuning in {'True', 'true', True, '1', 1}
    time = int(time)
    
    main(domain, instance, method, online, tuning, time)
    
