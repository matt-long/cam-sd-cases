import os

import dask
from dask.distributed import Client
from dask_jobqueue import PBSCluster

path_to_here = os.path.dirname(os.path.realpath(__file__))

USER = os.environ['USER']


def get_ClusterClient(memory='25GB', project='NCGD0011'):
    """get cluster and client"""
    cluster = PBSCluster(
        cores=1,
        memory=memory,
        processes=1,
        queue='casper',
        local_directory=f'/glade/scratch/{USER}/dask-workers',
        log_directory=f'/glade/scratch/{USER}/dask-workers',
        resource_spec=f'select=1:ncpus=1:mem={memory}',
        project=project,
        walltime='06:00:00',
        interface='ib0',
    )

    jupyterhub_server_name = os.environ.get('JUPYTERHUB_SERVER_NAME', None)
    dashboard_link = 'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'
    if jupyterhub_server_name:
        dashboard_link = (
            'https://jupyterhub.hpc.ucar.edu/stable/user/'
            + '{USER}'
            + f'/{jupyterhub_server_name}/proxy/'
            + '{port}/status'
        )
    dask.config.set({'distributed.dashboard.link': dashboard_link})
    client = Client(cluster)
    return cluster, client

