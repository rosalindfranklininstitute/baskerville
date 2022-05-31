import os, sys, time, signal, ctypes, io
import subprocess
import logging
import socket
from absl import app, flags
import pandas as pd

# Logging flags
flags.DEFINE_bool('log_nvsmi', False, help='')
flags.DEFINE_bool('log_env', False, help='')
FLAGS = flags.FLAGS

def task(argv, logger, MPI):

    import numpy as np

    # Do a broadcast from each node to the other nodes to test communication
    MPI.COMM_WORLD.barrier()
    for idx in range(MPI.COMM_WORLD.Get_size()):

        xs = -np.ones((MPI.COMM_WORLD.Get_size(),))
        if idx == MPI.COMM_WORLD.Get_rank():
            xs[:] = idx
            logger.info(f'BCAST ROOT {xs}')

        MPI.COMM_WORLD.barrier()
        logger.info(f'BEFORE BCAST {xs}')

        xs = MPI.COMM_WORLD.bcast(xs, root=idx)

        logger.info(f' AFTER BCAST {xs}')

    logger.info(f' AFTER MPI TEST')

    import jax
    import jax.numpy as jnp

    JAX_LOCAL_DEVICES = jax.local_devices()
    logger.debug(f'JAX Devices: {list(map(str, JAX_LOCAL_DEVICES))}')
    assert len(JAX_LOCAL_DEVICES) > 0

    # Create a JAX function to test local gpus
    @jax.jit
    def test_jax(xs):
        return xs @ xs.T

    # Create input array for this task instance
    xs = jnp.arange(MPI.COMM_WORLD.Get_size()) + MPI.COMM_WORLD.Get_rank()
    logger.info(f'BEFORE LOCAL JAX TEST | xs {xs.device()} {xs.shape} {xs}')

    # Run the JAX function which operates locally
    xs = test_jax(xs)
    logger.info(f' AFTER LOCAL JAX TEST | xs {xs.device()} {xs.shape} {xs}')

    log_nvidia_smi(logger)

    # Must import mpi4jax after jax
    import mpi4jax
    # Use cloned JAX communicator exclusively for JAX to ensure no deadlocks from
    # asynchronous execution compared to surrounding mpi4py communications.
    JAX_COMM_WORLD = MPI.COMM_WORLD.Clone()

    # Create a JAX function that will worth with mpi4jax without causing deadlocks
    @jax.jit
    def test_mpi4jax(xs):
        xs_sum, _ = mpi4jax.allreduce(xs, op=MPI.SUM, comm=JAX_COMM_WORLD)
        return xs_sum

    # Create input array for this task instance
    xs = jnp.arange(JAX_COMM_WORLD.Get_size()) + JAX_COMM_WORLD.Get_rank()
    logger.info(f'BEFORE JAX ALL-REDUCE-SUM | xs {xs.device()} {xs.shape} {xs}')

    # Run the JAX function which includes mpi4jax communication
    xs = test_mpi4jax(xs)
    logger.info(f' AFTER JAX ALL-REDUCE-SUM | xs {xs.device()} {xs.shape} {xs}')

    log_nvidia_smi(logger)

    # Create a JAX function to test local gpus
    @jax.pmap
    def test_jax_pmap(xs):
        return xs @ xs.T

    # Create input array for this task instance
    xs = jnp.arange(MPI.COMM_WORLD.Get_size()) + MPI.COMM_WORLD.Get_rank()
    xs = jax.device_put_replicated(xs, JAX_LOCAL_DEVICES)
    logger.info(f'BEFORE LOCAL JAX PMAP TEST | xs {xs.shape} {xs}')

    # Run the JAX function which operates locally
    xs = test_jax_pmap(xs)
    logger.info(f' AFTER LOCAL JAX PMAP TEST | xs {xs.shape} {xs}')

    log_nvidia_smi(logger)

    # Create a JAX function that will worth with mpi4jax without causing deadlocks
    @jax.pmap
    def test_mpi4jax_pmap(xs):
        xs_sum, _ = mpi4jax.allreduce(xs, op=MPI.SUM, comm=JAX_COMM_WORLD)
        return xs_sum

    # Create input array for this task instance
    xs = jnp.arange(JAX_COMM_WORLD.Get_size()) + JAX_COMM_WORLD.Get_rank()
    xs = jax.device_put_replicated(xs, JAX_LOCAL_DEVICES)
    logger.info(f'BEFORE JAX PMAP ALL-REDUCE-SUM | xs {xs.shape} {xs}')

    # Run the JAX function which includes mpi4jax communication
    xs = test_mpi4jax_pmap(xs)
    logger.info(f' AFTER JAX PMAP ALL-REDUCE-SUM | xs {xs.shape} {xs}')

    log_nvidia_smi(logger)

########################################################################################################################
# Everything below here is configuring logging output

def main(argv):

    # Kill child processes on exit
    def _set_pdeathsig(sig=signal.SIGTERM):
        def fn():
            return ctypes.CDLL("libc.so.6").prctl(1, sig)
        return fn

    # Decode the SLURM job information and the MPI rank of this task instance
    HOSTNAME = socket.gethostname()
    JOB_ID, JOB_NAME = os.environ['SLURM_JOB_ID'], os.environ['SLURM_JOB_NAME']
    WORLD_RANK, WORLD_SIZE = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])

    # Dump env and nvidia-smi output for this task before trying to do anything incase of failure
    if FLAGS.log_env:
        os.makedirs('logs/env', exist_ok=True)
        with open(f'logs/env/{JOB_ID}-{WORLD_RANK:04d}.env', 'w') as fp:
            subprocess.Popen(['env'], stdout=fp, stderr=fp).wait()
            fp.flush()

    # Start a background process to log nvidia-smi as a csv file for global performance monitoring
    if FLAGS.log_nvsmi:
        os.makedirs('logs/nvsmi', exist_ok=True)

        # Log human readable nvsmi
        with open(f'logs/nvsmi/{JOB_ID}-{WORLD_RANK:04d}.nvsmi', 'w') as fp:
            subprocess.Popen(['/usr/bin/nvidia-smi'], stdout=fp, stderr=fp).wait()
            fp.flush()

        # Set a background process logging nvsmi in csv format in 1 second intervals
        with open(f'logs/nvsmi/{JOB_ID}-{WORLD_RANK:04d}.csv', 'w') as fp:
            subprocess.Popen(['/usr/bin/nvidia-smi', '--format=csv', '--loop=1',
                              '--query-gpu=timestamp,uuid,power.draw,memory.used,memory.free,memory.total,'
                              'temperature.gpu,temperature.memory,utilization.gpu,utilization.memory'],
                             stdout=fp, preexec_fn=_set_pdeathsig())

    class MPIFilter(logging.Filter):
        # Filter to inject MPI rank and job information into logging output
        def filter(self, record):
            record.hostname = HOSTNAME
            record.job_id, record.job_name = JOB_ID, JOB_NAME
            record.world_rank, record.world_size = WORLD_RANK, WORLD_SIZE
            return True

    # Create a unique logger for this SLURM job for this MPI rank
    logging.getLogger().handlers.clear()
    logger = logging.getLogger(f'{JOB_ID}-{WORLD_RANK:04d}')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addFilter(MPIFilter())

    # Format the logging to include the SLURM job and MPI ranks
    formatter = logging.Formatter('%(asctime)s | %(hostname)s | %(job_id)s | '
                                  'W %(world_rank)03d:%(world_size)03d | '
                                  '%(levelname)10s | %(message)s')

    # Mirror logging to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Mirror logging to unique output file for this SLURM job and MPI rank
    file_handler = logging.FileHandler(f'logs/{JOB_ID}-{WORLD_RANK:04d}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Debug environment for this task instance
    CUDA_VISIBLE_DEVICES = sorted(map(int, filter(lambda y: len(y),
                           os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))))
    logger.info(f'CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES}')
    assert len(CUDA_VISIBLE_DEVICES) == len(set(CUDA_VISIBLE_DEVICES))

    log_nvidia_smi(logger)

    try:
        logger.debug(f'Initializing MPI')
        from mpi4py import MPI
        MPI.COMM_WORLD.barrier()

        logger.debug('Starting task...')
        task(argv=argv, logger=logger, MPI=MPI)
        time.sleep(5)
        logger.debug('Halting task...')

        logger.debug(f'Finalizing MPI')
        MPI.COMM_WORLD.barrier()
        MPI.Finalize()

    except Exception as ex:
        # Catch any top level exceptions and ensure they are logged
        logger.exception(ex, exc_info=True)

def log_nvidia_smi(logger):
    # Utility to get the human readable output of nvidia-smi for local logging
    proc = subprocess.Popen(['/usr/bin/nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.communicate()[0].decode("utf-8").strip().split('\n'):
        logger.info(f'nvidia-smi | {line}')

    logger.info(f'nvidia-smi |')

    proc = subprocess.Popen(['/usr/bin/nvidia-smi', '--format=csv', '--query-gpu=uuid,name'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc_stdout = proc.communicate()[0].decode("utf-8").strip()

    try:
        for line in str(pd.read_csv(io.StringIO(proc_stdout))).split('\n'):
            logger.info(f'nvidia-smi | {line}')
    except Exception as ex:
        for line in proc_stdout.split('\n'):
            logger.info(f'nvidia-smi | {line}')
        logger.exception(ex, exc_info=True)

if __name__ == "__main__":
    app.run(main)
