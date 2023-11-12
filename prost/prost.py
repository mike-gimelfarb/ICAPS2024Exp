import configparser
import os
import sys

from pyRDDLGym.Core.Policies.RDDLSimAgent import RDDLSimAgent
from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

# read the command line args
args = sys.argv
if len(args) != 4:
    print('')
    print('    Usage: python prost.py <domain> <instance> <time>')
    print('')
    sys.exit(1)
domain, instance, time = args[1], args[2], int(args[3])

# extract the number of rounds from config
config = configparser.RawConfigParser()
config.optionxform = str 
config.read('global.cfg')
rounds = int(config.get('General', 'episodes'))

# build the environment
EnvInfo = RDDLRepoManager().get_problem(domain)
domain_path = EnvInfo.get_domain()
instance_path = EnvInfo.get_instance(instance)

# launch the RDDL server
agent = RDDLSimAgent(domain_path, instance_path, rounds, 99999)
agent.run()
agent.dump_data(os.path.join(os.environ.get('PROST_OUT'), 
                f'data_{domain}_{instance}_prost_True_{time}.json'))