from datetime import timedelta
import os
import sys

from pyRDDLGym.Core.Env.RDDLEnv import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import _parse_config_file

from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager
    

def esttime(domain, instance, tuning, time):
    EnvInfo = RDDLRepoManager().get_problem(domain)   
    env = RDDLEnv(domain=EnvInfo.get_domain(),
                  instance=EnvInfo.get_instance(instance),
                  enforce_action_constraints=True)
    env.set_visualizer(None)
    
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    _, global_args = _parse_config_file(os.path.join(ROOT_PATH, 'baselines', 'global.cfg'))
    
    time_per_episode = time * env.horizon
    total_time = 0
    if tuning:
        total_time += global_args['rounds'] * global_args['trials'] * time_per_episode
    total_time += global_args['episodes'] * time_per_episode
    
    for f in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        total_time_f = timedelta(seconds=int(total_time * f))
        print(f'time @ {f}: {total_time_f}')        


if __name__ == '__main__':
    args = sys.argv
    #domain, instance, tuning, time = args[1:7]
    domain, instance, tuning, time = 'UAV_ippc2023', 1, True, 1
    domain = str(domain)
    instance = str(instance)
    tuning = tuning in {'True', 'true', True, '1', 1}
    time = int(time)
    
    esttime(domain, instance, tuning, time)
