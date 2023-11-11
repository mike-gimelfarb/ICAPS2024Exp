import json
import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import _parse_config_file, _load_config
from pyRDDLGym.Core.Policies.Agents import NoOpAgent, RandomAgent

from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager
    

def main(domain, instance, method, online, tuning, time):
    print('receiving args...', flush=True)
    print(f'args = {domain}, {instance}, {method}, {online}, {tuning}, {time}', flush=True)
        
    # load global config
    print('loading global config...', flush=True)
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    _, global_args = _parse_config_file(os.path.join(ROOT_PATH, 'baselines', 'global.cfg'))
    print('loading global config complete!', flush=True)
    
    # path where to log outputs
    outputpath = os.path.join(ROOT_PATH, 'outputs', method,
                              f'{domain}_{instance}_{method}_{online}_{time}')
    
    # create the environment
    EnvInfo = RDDLRepoManager().get_problem(domain)   
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(instance),
                  enforce_action_constraints=True,
                  log=True, log_path=outputpath)
    
    # load the config file with planner settings
    print('loading config...', flush=True)
    config, args = _parse_config_file(os.path.join(ROOT_PATH, 'configs', f'{domain}.cfg'))
    
    # override default config settings here
    args['train_seconds'] = time * (1 if online else env.horizon)
    if not online:
        args['rollout_horizon'] = None
    planner_args, plan_args, train_args = _load_config(config, args)
    print('loading config complete!', flush=True)
    
    # dispatch to policy creation method
    print('begin policy creation...', flush=True)
    if method == 'jaxplan':
        from baselines.jaxplan import jax_policy
        policy = jax_policy(env, online, tuning,
                            config, args, planner_args, plan_args, train_args,
                            outputpath, global_args)
        
    elif method == 'gurobiplan':
        from baselines.gurobiplan import gurobi_policy
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
    print('policy creation complete!', flush=True)
    
    # evaluation
    print('begin policy evaluation...', flush=True)
    result = policy.evaluate(env, verbose=False, episodes=global_args['episodes'])
    print('policy evaluation complete!', flush=True)
    
    # dump all history to files
    print('writing logs...', flush=True)
    with open(outputpath + '.json', 'w') as fp:
        json.dump(result, fp, indent=4)
    env.close()
    print('writing logs complete!', flush=True)


if __name__ == '__main__':
    args = sys.argv
    domain, instance, method, online, tuning, time = args[1:7]
    domain = str(domain)
    instance = str(instance)
    online = online in {'True', 'true', True, '1', 1}
    tuning = tuning in {'True', 'true', True, '1', 1}
    time = int(time)
    
    main(domain, instance, method, online, tuning, time)
    
