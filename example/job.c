#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define MAX_HEADER_LEN 256

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int WORLD_RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_SIZE;
    MPI_Comm_rank(MPI_COMM_WORLD, &WORLD_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &WORLD_SIZE);
    MPI_Comm comm_local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_local);
    MPI_Comm_rank(comm_local, &LOCAL_RANK);
    MPI_Comm_size(comm_local, &LOCAL_SIZE);

    char LOG_HEADER[MAX_HEADER_LEN];
    sprintf(LOG_HEADER, "W %03d:%03d | L %03d:%03d", WORLD_RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_SIZE);

    printf("%s | Starting...\n", LOG_HEADER);
    MPI_Barrier(MPI_COMM_WORLD);

    int *xs = (int*) malloc(sizeof(int) * WORLD_SIZE);

    for (int i = 0; i < WORLD_SIZE; i++) {
        if (i == WORLD_RANK) {
            printf("%s | BCAST ROOT", LOG_HEADER);
            for (int j = 0; j < WORLD_SIZE; j++) {
                xs[j] = WORLD_RANK;
                printf(" %d", xs[j]);
            }
            printf("\n");
        }

        MPI_Bcast(xs, WORLD_SIZE, MPI_INT, i, MPI_COMM_WORLD);

        for (int j = 0; j < WORLD_SIZE; j++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (j == WORLD_RANK) {
                printf("%s | BCAST", LOG_HEADER);
                for (int k = 0; k < WORLD_SIZE; k++) {
                    xs[k] = WORLD_RANK;
                    printf(" %d", xs[k]);
                }
                printf("\n");
            }
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    free(xs);

    printf("%s | Halting...\n", LOG_HEADER);

    MPI_Finalize();
    return 0;
}
