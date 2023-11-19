from datetime import datetime

from pyRDDLGym.Core.Gurobi.GurobiParameterTuning import GurobiParameterTuningReplan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiStraightLinePlan
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOfflineController
from pyRDDLGym.Core.Gurobi.GurobiRDDLPlanner import GurobiOnlineController


def gurobi_policy(env, online, tuning, args, outputpath, global_args):
    
    # run hyper-parameter tuning
    T = args['rollout_horizon']
    if tuning and online:
        lookahead_range = list(range(1, 21)) + [22, 24, 26, 28, 30, 35, 40]
        tuning = GurobiParameterTuningReplan(env,
                                             lookahead_range=lookahead_range,
                                             timeout_training=args['train_seconds'],
                                             eval_trials=global_args['trials'],
                                             num_workers=global_args['batch'])        
        T = tuning.tune(key=int(datetime.now().timestamp()),
                        filename=outputpath + '_gp')
        
    # solve planning problem with new optimal parameters
    model_params = {'OutputFlag': 0, 'NonConvex': 2, 'TimeLimit': args['train_seconds']}
    if online: 
        return GurobiOnlineController(rddl=env.model,
                                      plan=GurobiStraightLinePlan(),
                                      rollout_horizon=T,
                                      model_params=model_params,
                                      verbose=args['verbose'])
    else:
        return GurobiOfflineController(rddl=env.model,
                                       plan=GurobiStraightLinePlan(),
                                       model_params=model_params,
                                       verbose=args['verbose'])
