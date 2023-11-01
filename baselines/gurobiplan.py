from datetime import datetime

from pyRDDLGym.Core.Gurobi.GurobiParameterTuning import GurobiParameterTuningReplan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiStraightLinePlan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOfflineController
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOnlineController


def gurobi_policy(env, online, do_tune, args, outputpath, global_args):
    
    # run hyper-parameter tuning
    params = {'T': args['rollout_horizon']}
    if do_tune and online:
        tuning = GurobiParameterTuningReplan(env,
                                             timeout_training=args['train_seconds'],
                                             num_workers=global_args['gp_cpus_gurobi'],
                                             gp_iters=global_args['gp_iters_gurobi'])
        tuning.hyperparams_dict['T'] = (1, env.horizon, int)
        
        params = tuning.tune(key=int(datetime.now().timestamp()), 
                             filename=outputpath + '_gp')
        
    # solve planning problem with new optimal parameters
    model_params = {'OutputFlag': int(not online),
                    'NonConvex': 2,
                    'TimeLimit': args['train_seconds']}
    if online: 
        return GurobiOnlineController(rddl=env.model,
                                      plan=GurobiStraightLinePlan(),
                                      rollout_horizon=params['T'],
                                      model_params=model_params)
    else:
        return GurobiOfflineController(rddl=env.model,
                                       plan=GurobiStraightLinePlan(),
                                       model_params=model_params)
