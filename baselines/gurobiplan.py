from datetime import datetime

from pyRDDLGym.Core.Gurobi.GurobiParameterTuning import GurobiParameterTuningReplan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiStraightLinePlan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOfflineController
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOnlineController


def gurobi_policy(env, online, tuning, args, outputpath, global_args):
    
    # run hyper-parameter tuning
    params = {'T': args['rollout_horizon']}
    if tuning and online:
        tuning = GurobiParameterTuningReplan(env,
                                             timeout_training=args['train_seconds'],
                                             num_workers=global_args['gp_cpus_gurobi'],
                                             gp_iters=global_args['gp_iters_gurobi'])
        
        params = tuning.tune(key=int(datetime.now().timestamp()), 
                             filename=outputpath + '_gp')
        
    # solve planning problem with new optimal parameters
    model_params = {'OutputFlag': 0, 'NonConvex': 2, 'TimeLimit': args['train_seconds']}
    if online: 
        return GurobiOnlineController(rddl=env.model,
                                      plan=GurobiStraightLinePlan(),
                                      rollout_horizon=params['T'],
                                      model_params=model_params)
    else:
        return GurobiOfflineController(rddl=env.model,
                                       plan=GurobiStraightLinePlan(),
                                       model_params=model_params)
