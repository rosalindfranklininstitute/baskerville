import os, sys, time, signal, ctypes
import subprocess
import logging
import socket
from absl import app, flags

flags.DEFINE_string('train_dataset', None, help='')
flags.mark_flag_as_required('train_dataset')
flags.DEFINE_string('val_dataset', None, help='')
flags.mark_flag_as_required('val_dataset')

flags.DEFINE_integer('batch_size', 10000, help='')
flags.DEFINE_integer('val_batch_size', 10000, help='')

flags.DEFINE_float('learning_rate', 0.1, help='')

flags.DEFINE_integer('random_seed', 42, help='')
flags.DEFINE_integer('log_every', 100, help='')
flags.DEFINE_integer('epochs', 90, help='')

flags.DEFINE_bool('log_nvsmi', False, help='')
flags.DEFINE_bool('log_env', False, help='')
FLAGS = flags.FLAGS

def task(argv, logger):

    logger.info(f'Importing JAX')

    # Must wait to import jax until after CUDA_VISIBLE_DEVICES is set correctly
    import jax
    import jax.numpy as jnp
    import numpy as np

    logger.info(f'Initializing MPI')

    from mpi4py import MPI
    import mpi4jax
    MPI_COMM_WORLD = MPI.COMM_WORLD
    JAX_COMM_WORLD = MPI_COMM_WORLD.Clone()

    import hub

    ds_train = hub.load(FLAGS.train_dataset, read_only=True, memory_cache_size=8192, local_cache_size=150000)
    ds_val   = hub.load(FLAGS.val_dataset,   read_only=True, memory_cache_size=8192, local_cache_size=150000)

    X = ds_train.images[0].numpy()
    Y = ds_train.labels[0].numpy()
    logger.info(f'X train {X.shape} {X.dtype} {X.min()} {X.max()}')
    logger.info(f'Y train {Y.shape} {Y.dtype}')

    X = ds_val.images[0].numpy()
    Y = ds_val.labels[0].numpy()
    logger.info(f'X val {X.shape} {X.dtype} {X.min()} {X.max()}')
    logger.info(f'Y val {Y.shape} {Y.dtype}')

    logger.info(f'Finalizing MPI')

    MPI_COMM_WORLD.barrier()
    MPI.Finalize()

def main(argv):

    # Kill child processes on exit
    def _set_pdeathsig(sig=signal.SIGTERM):
        def fn():
            return ctypes.CDLL("libc.so.6").prctl(1, sig)
        return fn

    # Decode the SLURM job information and the MPI rank of this task instance
    HOSTNAME = socket.gethostname()
    JOB_ID, JOB_NAME = os.environ['SLURM_JOB_ID'], os.environ['SLURM_JOB_NAME']
    LOCAL_RANK, LOCAL_SIZE = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']), int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    WORLD_RANK, WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_RANK']), int(os.environ['OMPI_COMM_WORLD_SIZE'])

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
        with open(f'logs/nvsmi/{JOB_ID}-{WORLD_RANK:04d}.nvsmi.csv', 'w') as fp:
            subprocess.Popen(['/usr/bin/nvidia-smi', '--format=csv', '--loop=1',
                              '--query-gpu=timestamp,uuid,power.draw,memory.used,memory.free,memory.total,'
                              'temperature.gpu,temperature.memory,utilization.gpu,utilization.memory'],
                             stdout=fp, preexec_fn=_set_pdeathsig())

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
        CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES[(LOCAL_RANK * GPUS_PER_TASK):((LOCAL_RANK + 1) * GPUS_PER_TASK)]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, CUDA_VISIBLE_DEVICES))
        logger.info(f'Setting CUDA_VISIBLE_DEVICES to {CUDA_VISIBLE_DEVICES}')

    log_nvidia_smi(logger)

    try:
        logger.debug('Starting...')
        task(argv=argv, logger=logger)
        time.sleep(10)
        logger.debug('Halting...')
    except Exception as ex:
        # Catch any top level exceptions and ensure they are logged
        logger.exception(ex, exc_info=True)

def log_nvidia_smi(logger):
    # Utility to get the human readable output of nvidia-smi for local logging
    proc = subprocess.Popen(['/usr/bin/nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.communicate()[0].decode("utf-8").strip().split('\n'):
        logger.info(f'nvidia-smi | {line}')

if __name__ == "__main__":
    app.run(main)
