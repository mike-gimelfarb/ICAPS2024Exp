FROM python:3.10

WORKDIR /usr/src/RDDL-demo-agent

# copy all the files to the container
COPY . .

# install dependencies
RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade git+https://github.com/ataitler/pyRDDLGym
RUN pip install --upgrade git+https://github.com/ataitler/rddlrepository@icaps-fixed

# entry point for the docker, this will always run with the docker
# must be FULL PATH for singularity compatibility
ENTRYPOINT ["python", "/usr/src/RDDL-demo-agent/main.py"]

# run the command, this serves as default args to the entrypoint
CMD []



