from datetime import datetime
import jax

from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLP
from pyRDDLGym.Core.Jax.JaxParameterTuning import JaxParameterTuningSLPReplan
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import _load_config
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxRDDLBackpropPlanner
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOfflineController
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxOnlineController


def jax_policy(env, online, tuning,
               config, args, planner_args, plan_args, train_args,
               outputpath, global_args):
    
    # run hyper-parameter tuning
    if tuning:
        if online:
            tuning_obj = JaxParameterTuningSLPReplan 
        else:
            tuning_obj = JaxParameterTuningSLP
        
        # create tuning instance with horizon range from environment
        tuning = tuning_obj(env=env,
                            train_epochs=9999999,
                            timeout_training=args['train_seconds'],
                            timeout_tuning=global_args['total_time'],
                            planner_kwargs=planner_args,
                            plan_kwargs=plan_args,
                            num_workers=global_args['cpus_jax'],
                            gp_iters=9999999)
        
        params = tuning.tune(key=jax.random.PRNGKey(int(datetime.now().timestamp())),
                             filename=outputpath + '_gp')
                
        # update config with optimal hyper-parameters
        args['method_kwargs']['initializer'] = 'normal'
        args['method_kwargs']['initializer_kwargs'] = {'stddev': params['std']}
        args['optimizer_kwargs']['learning_rate'] = params['lr']
        args['logic_kwargs']['weight'] = params['w']
        args['policy_hyperparams'] = {k: params['wa'] for k in args['policy_hyperparams']}
        if online:
            args['rollout_horizon'] = params['T']       
        planner_args, plan_args, train_args = _load_config(config, args)
            
    # solve planning problem with new optimal parameters
    planner = JaxRDDLBackpropPlanner(rddl=env.model, **planner_args)    
    if online:
        return JaxOnlineController(planner, **train_args)
    else:
        return JaxOfflineController(planner, train_on_reset=True, **train_args)

