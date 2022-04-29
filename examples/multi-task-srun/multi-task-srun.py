import os, sys, time
import subprocess
import logging
import socket

# Decode the SLURM job information and the MPI rank of this task instance
HOSTNAME = socket.gethostname()
JOB_ID, JOB_NAME = os.environ['SLURM_JOB_ID'], os.environ['SLURM_JOB_NAME']
WORLD_RANK, WORLD_SIZE = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])

# Dump env and nvidia-smi output for this task before trying to do anything incase of failure
with open(f'{JOB_ID}-{WORLD_RANK:04d}.env', 'w') as fp:
    subprocess.Popen(['env'], stdout=fp, stderr=fp).wait()
    fp.flush()
with open(f'{JOB_ID}-{WORLD_RANK:04d}.nvsmi', 'w') as fp:
    subprocess.Popen(['/usr/bin/nvidia-smi'], stdout=fp, stderr=fp).wait()
    fp.flush()

# Start a background process to log nvidia-smi as a csv file for global performance monitoring
with open(f'{JOB_ID}-{WORLD_RANK:04d}.nvsmi.csv', 'w') as fp:
    subprocess.Popen(['/usr/bin/nvidia-smi', '--format=csv', '--loop=1',
                      '--query-gpu=timestamp,uuid,power.draw,memory.used,memory.free,memory.total,'
                                  'temperature.gpu,temperature.memory,utilization.gpu,utilization.memory'], stdout=fp)

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
logger.addFilter(MPIFilter())

# Format the logging to include the SLURM job and MPI ranks
formatter = logging.Formatter('%(asctime)s | %(world_rank)03d:%(world_size)03d | %(hostname)s | '
                              '%(job_id)s | %(job_name)s | %(levelname)10s | %(message)s')

# Mirror logging to stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# Mirror logging to unique output file for this SLURM job and MPI rank
file_handler = logging.FileHandler(f'{JOB_ID}-{WORLD_RANK:04d}.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def nvidia_smi():
    # Utility to get the human readable output of nvidia-smi for local logging
    proc = subprocess.Popen(['/usr/bin/nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.communicate()[0].decode("utf-8").strip()

########################################################################################################################

try:
    logger.debug('Starting...')

    # Debug environment for this task instance
    logger.debug(f'PATH {os.environ.get("PATH", "")}')
    logger.debug(f'LD_LIBRARY_PATH {os.environ.get("LD_LIBRARY_PATH", "")}')

    CUDA_VISIBLE_DEVICES = sorted(map(int, filter(lambda y: len(y), os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))))
    logger.info(f'CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES}')
    assert len(CUDA_VISIBLE_DEVICES) == len(set(CUDA_VISIBLE_DEVICES))

    for line in nvidia_smi().split('\n'):
        logger.info(f'nvidia-smi | {line}')

    ####################################################################################################################

    # Must wait to import jax until after CUDA_VISIBLE_DEVICES is srt correctly
    import jax
    import jax.numpy as jnp
    import numpy as np

    # Create a JAX function to test local gpu
    @jax.jit
    def test_jax(xs):
       return xs @ xs.T

    # Create input array for this task instance
    xs = jnp.arange(WORLD_SIZE) + WORLD_RANK
    logger.info(f'BEFORE LOCAL MATMUL | xs {xs.device()} {xs.shape} {xs}')

    # Run the JAX function which operates locally
    xs = test_jax(xs)
    logger.info(f' AFTER LOCAL MATMUL | xs {xs.device()} {xs.shape} {xs}')

    # Log nvidia-smi to ensure task instances got the correct GPU bindings
    for line in nvidia_smi().split('\n'):
        logger.info(f'nvidia-smi | {line}')

except Exception as ex:
    # Catch any top level exceptions and ensure they are logged
    logger.exception(ex, exc_info=True)

# Keep process alive for a bit at the end of the job to ensure nvidia-smi process binding is reported accurately
time.sleep(10)
logger.debug('Halting...')
