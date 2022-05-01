import os, sys, time
import subprocess
import logging
import socket

# Decode the SLURM job information and the MPI rank of this task instance
HOSTNAME = socket.gethostname()
JOB_ID, JOB_NAME = os.environ['SLURM_JOB_ID'], os.environ['SLURM_JOB_NAME']
LOCAL_RANK, LOCAL_SIZE = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']), int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
WORLD_RANK, WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_RANK']), int(os.environ['OMPI_COMM_WORLD_SIZE'])

# Dump env and nvidia-smi output for this task before trying to do anything incase of failure
if "--log-env" in sys.argv:
    sys.argv.remove("--log-env")
    os.makedirs('logs/env', exist_ok=True)
    with open(f'logs/env/{JOB_ID}-{WORLD_RANK:04d}.env', 'w') as fp:
        subprocess.Popen(['env'], stdout=fp, stderr=fp).wait()
        fp.flush()

# Start a background process to log nvidia-smi as a csv file for global performance monitoring
if "--log-nvsmi" in sys.argv:
    sys.argv.remove("--log-nvsmi")
    os.makedirs('logs/nvsmi', exist_ok=True)

    # Log human readable nvsmi
    with open(f'logs/nvsmi/{JOB_ID}-{WORLD_RANK:04d}.nvsmi', 'w') as fp:
        subprocess.Popen(['/usr/bin/nvidia-smi'], stdout=fp, stderr=fp).wait()
        fp.flush()

    # Set a background process logging nvsmi in csv format in 1 second intervals
    with open(f'logs/nvsmi/{JOB_ID}-{WORLD_RANK:04d}.nvsmi.csv', 'w') as fp:
        subprocess.Popen(['/usr/bin/nvidia-smi', '--format=csv', '--loop=1',
                          '--query-gpu=timestamp,uuid,power.draw,memory.used,memory.free,memory.total,'
                                       'temperature.gpu,temperature.memory,utilization.gpu,utilization.memory'], stdout=fp)

class MPIFilter(logging.Filter):
    # Filter to inject MPI rank and job information into logging output
    def filter(self, record):
        record.hostname = HOSTNAME
        record.job_id, record.job_name = JOB_ID, JOB_NAME
        record.local_rank, record.local_size = LOCAL_RANK, LOCAL_SIZE
        record.world_rank, record.world_size = WORLD_RANK, WORLD_SIZE
        return True

# Create a unique logger for this SLURM job for this MPI rank
logging.getLogger().handlers.clear()
logger = logging.getLogger(f'{JOB_ID}-{WORLD_RANK:04d}')
logger.setLevel(logging.DEBUG)
logger.addFilter(MPIFilter())

# Format the logging to include the SLURM job and MPI ranks
formatter = logging.Formatter('%(asctime)s | %(hostname)s | %(job_id)s | '
                              'W %(world_rank)03d:%(world_size)03d | '
                              'L %(local_rank)03d:%(local_size)03d | '
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

    if len(CUDA_VISIBLE_DEVICES) > 0:
        # Assert that there are an even number of GPUs for local tasks on this node
        assert (len(CUDA_VISIBLE_DEVICES) % LOCAL_SIZE) == 0

        # Determine the GPUs that should be used for this local task instance
        GPUS_PER_TASK = len(CUDA_VISIBLE_DEVICES) // LOCAL_SIZE
        CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES[(LOCAL_RANK*GPUS_PER_TASK):((LOCAL_RANK+1)*GPUS_PER_TASK)]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, CUDA_VISIBLE_DEVICES))
        logger.info(f'Setting CUDA_VISIBLE_DEVICES to {CUDA_VISIBLE_DEVICES}')

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
    logger.info(f'BEFORE LOCAL JAX TEST | xs {xs.device()} {xs.shape} {xs}')

    # Run the JAX function which operates locally
    xs = test_jax(xs)
    logger.info(f' AFTER LOCAL JAX TEST | xs {xs.device()} {xs.shape} {xs}')

    # Log nvidia-smi to ensure task instances got the correct GPU bindings
    for line in nvidia_smi().split('\n'):
        logger.info(f'nvidia-smi | {line}')

    ####################################################################################################################

    logger.info(f'BEFORE MPI TEST')

    # Initialize MPI
    from mpi4py import MPI
    MPI_COMM_WORLD = MPI.COMM_WORLD

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

    ####################################################################################################################

    # Must import mpi4jax after jax
    # Use cloned JAX communicator exclusively for JAX to ensure no deadlocks from
    # asynchronous execution compared to surrounding mpi4py communications.
    import mpi4jax
    JAX_COMM_WORLD = MPI_COMM_WORLD.Clone()

    # TODO add pmap to example function to utilize multiple local GPUs (currently only uses GPU 0)
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

except Exception as ex:
    # Catch any top level exceptions and ensure they are logged
    logger.exception(ex, exc_info=True)

# Keep process alive for a bit at the end of the job to ensure nvidia-smi process binding is reported accurately
time.sleep(10)
logger.debug('Halting...')
