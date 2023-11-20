import json
import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import _parse_config_file, _load_config
from pyRDDLGym.Core.Policies.Agents import NoOpAgent, RandomAgent

from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

DOMAIN_LIST = [
    'AcademicAdvising_MDP_ippc2014',
    'CooperativeRecon_MDP_ippc2011',
    'CrossingTraffic_MDP_ippc2011',
    'Elevators_MDP_ippc2011',
    'GameOfLife_MDP_ippc2011',
    'HVAC_ippc2023',
    'MarsRover_ippc2023',
    'MountainCar_ippc2023',
    'Navigation_MDP_ippc2011',
    'PowerGen_ippc2023',
    'RaceCar_ippc2023',
    'Reservoir_ippc2023',
    'SkillTeaching_MDP_ippc2011',
    'SysAdmin_MDP_ippc2011',
    'Tamarisk_MDP_ippc2014',
    'Traffic_MDP_ippc2011',
    'TriangleTireworld_MDP_ippc2014',
    'UAV_ippc2023',
    'Wildfire_MDP_ippc2014'
]


def main(domain, instance, method, online, tuning, time):
    domain_num = domain
    domain = DOMAIN_LIST[domain_num - 1]
    print(f'args = {domain}, {instance}, {method}, {online}, {tuning}, {time}\n', flush=True)
    
    # load global config
    print('loading global config...', flush=True)
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    _, global_args = _parse_config_file(os.path.join(ROOT_PATH, 'global.cfg'))
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
        from jaxplan import jax_policy
        policy = jax_policy(env, online, tuning,
                            config, args, planner_args, plan_args, train_args,
                            outputpath, global_args)
        
    elif method == 'gurobiplan':
        from gurobiplan import gurobi_policy
        policy = gurobi_policy(env, online, tuning, args, outputpath, global_args)
    
    elif method == 'noop':
            policy = NoOpAgent(action_space=env.action_space,
                               num_actions=env.numConcurrentActions)
            
    elif method == 'random':
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
    args = sys.argv[1:]
    if len(args) != 6:
        print('')
        print('    usage main.py <domain> <instance> <method> <online> <tuning> <time>')
        print('')
        sys.exit(1)
    domain, instance, method, online, tuning, time = args[:6]
    domain = int(domain)
    assert domain >= 0
    instance = str(instance)
    online = online in {'True', 'true', True, '1', 1}
    tuning = tuning in {'True', 'true', True, '1', 1}
    time = int(time)
    
    main(domain, instance, method, online, tuning, time)
    
