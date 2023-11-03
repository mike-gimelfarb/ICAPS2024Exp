# python run.py dom.rddl prob.rddl <rounds (e.g., 30)> <time (e.g., 300)>

import sys

from pyRDDLGym.Core.Policies.RDDLSimAgent import RDDLSimAgent
from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

args = sys.argv
if len(args) != 5:
    print('Usage: python run.py dom.rddl prob.rddl <rounds> <time>')
    sys.exit(1)
    
domain, instance, rounds, time = args[1], args[2], int(args[3]), int(args[4])

EnvInfo = RDDLRepoManager().get_problem(domain)
domain_path = EnvInfo.get_domain()
instance_path = EnvInfo.get_instance(instance)

agent = RDDLSimAgent(domain_path, instance_path, rounds, time)
agent.run()
agent.dump_data(f'/workspace/data.json')