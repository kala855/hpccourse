// Copyright 2011 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// An intro MPI hello world program that uses MPI_Init, MPI_Comm_size,
// MPI_Comm_rank, MPI_Finalize, and MPI_Get_processor_name.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "someDefinitions.h"

int main(int argc, char** argv) {
  // Initialize the MPI environment. The two arguments to MPI Init are not
  // currently used by MPI implementations, but are there in case future
  // implementations might need the arguments.
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if(world_size < 2){
      fprintf(stderr, "World size must be equal to 2 for %s\n", argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  float *number;
  number = (float*) malloc(1*sizeof(float));
  if(world_rank == 0){
      printf("Running on %d nodes \n", world_size);
      number[0] = 5.0;
      MPI_Send(number, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
  }else if(world_rank == 1){
      MPI_Recv(number, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      cudaCall(world_rank, number);
      MPI_Send(number,1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }

  if (world_rank==0){
      MPI_Recv(number, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("The result is %f\n", number[0]);
  }

  // Print off a hello world message
//  printf("Hello world from processor %s, rank %d out of %d processors\n",
 //        processor_name, world_rank, world_size);

  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();
}
