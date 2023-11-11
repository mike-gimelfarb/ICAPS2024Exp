from datetime import timedelta
import os
import sys

from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import _parse_config_file
    

def esttime(horizon=200, tuning=True, time=1, margin=1):
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    _, global_args = _parse_config_file(os.path.join(ROOT_PATH, 'baselines', 'global.cfg'))
    
    time_per_episode = time * horizon
    total_time = 0
    if tuning:
        total_time += global_args['rounds'] * global_args['trials'] * time_per_episode
    total_time += global_args['episodes'] * time_per_episode
    
    total_time_f = timedelta(seconds=int(total_time * margin))
    return total_time_f


if __name__ == '__main__':
    args = sys.argv[1:]
    horizon, tuning, time, margin = args[:4]
    horizon = int(horizon)
    tuning = tuning in {'True', 'true', True, '1', 1}
    time = int(time)
    margin = float(margin)
    
    print(esttime(horizon, tuning, time, margin))
