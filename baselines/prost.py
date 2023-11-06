# python run.py dom.rddl prob.rddl <rounds (e.g., 30)> <time (e.g., 300)>

import configparser
import sys

from pyRDDLGym.Core.Policies.RDDLSimAgent import RDDLSimAgent
from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

args = sys.argv
if len(args) != 4:
    print('Usage: python run.py <domain> <instance> <time>')
    sys.exit(1)
domain, instance, time = args[1], args[2], int(args[3])

config = configparser.RawConfigParser()
config.optionxform = str 
config.read('global.cfg')
rounds = int(config.get('General', 'episodes'))
    
EnvInfo = RDDLRepoManager().get_problem(domain)
domain_path = EnvInfo.get_domain()
instance_path = EnvInfo.get_instance(instance)

agent = RDDLSimAgent(domain_path, instance_path, rounds, time)
agent.run()
agent.dump_data(f'/workspace/data.json')
