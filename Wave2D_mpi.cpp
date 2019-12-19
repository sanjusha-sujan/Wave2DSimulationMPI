/*
 * Wave2D_mpi.cpp
 *
 *  Created on: Oct 19, 2019
 *      Author: sanjushacheemakurthi
 */

#include <iostream>
#include <sys/resource.h>
#include "Timer.h"
#include <stdlib.h>   // atoi
#include <algorithm>
#include <math.h>
#include <memory.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>

int default_size = 100;  // the default system size
int defaultCellWidth = 8;
double c = 1.0;      // wave speed
double dt = 0.1;     // time quantum
double dd = 2.0;     // change in system

using namespace std;

int main(int argc, char *argv[]) {

  // verify arguments
  if (argc != 5) {
    cerr << "usage: Wave2D size max_time interval" << endl;
    return -1;
  }
  int size = atoi(argv[1]);
  int max_time = atoi(argv[2]);
  int interval = atoi(argv[3]);
  int num_threads = atoi(argv[4]);

  if (size < 2 || max_time < 3 || interval < 0) {
    cerr << "usage: Wave2D size max_time interval" << endl;
    cerr << "       where size >= 100 && time >= 3 && interval >= 0" << endl;
    return -1;
  }

  int my_rank = 0;            // used by MPI
  int mpi_size = 1;           // used by MPI

  omp_set_num_threads(num_threads);

  double z[3][size][size];

#pragma omp parallel for default(none) shared(z,size)
  for (int p = 0; p < 3; p++) {
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        z[p][i][j] = 0.0;  // no wave
      }
    }
  }

//  int provided;
//
//  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);  // start MPI
//
//  if (provided < MPI_THREAD_FUNNELED) {
//    cout << "Not a high enough level of thread support!" << endl;
//    MPI_Abort(MPI_COMM_WORLD, 1);
//  }
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  // start a timer
  Timer time;
  time.start();
  int weight = size / default_size;

  // Calculating the wave equation for t=0

#pragma omp parallel for default(none) shared(z,weight,size)
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (i > 40 * weight && i < 60 * weight && j > 40 * weight && j < 60 * weight) {
        z[0][i][j] = 20.0;
      } else {
        z[0][i][j] = 0.0;
      }
    }
  }

  double pow_1_calculation = ((pow(c, 2) / 2) * pow((dt / dd), 2));

  // Calculating the wave equation for t=1
#pragma omp parallel
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (i == 0 || i == size - 1 || j == 0 || j == size - 1) {
        z[1][i][j] = 0.0;
      } else {

        z[1][i][j] = z[0][i][j]
            + pow_1_calculation
                * (z[0][i + 1][j] + z[0][i - 1][j] + z[0][i][j + 1] + z[0][i][j - 1] - 4.0 * z[0][i][j]);
      }
    }
  }
//Calculating the stripe size
  int stripe = size / mpi_size;

  cerr << "rank[" << my_rank << "]" << " range = " << stripe * my_rank << " ~ " << stripe * (my_rank + 1) - 1 << endl;

  int zeroth_index = 0;
  int first_index = 1;
  int second_index = 2;
  double (*zeroth_index_pointer)[size][size] = &z[0];
  double (*first_index_pointer)[size][size] = &z[1];
  double (*second_index_pointer)[size][size] = &z[2];

  double pow_2_calculation = (pow(c, 2) * pow((dt / dd), 2));

  /**
   *  Calculating wave equation from t=2..max-time.
   */
#pragma omp parallel for
  for (int t = 2; t < max_time; t++) {


    /**
     *  Neighbor information exchange.
     *  Even ranks will send first and then receive info required from the neighbors.
     *  Odd ranks will receive first and then send info required from the neighbors.
     */
    if (t > 2) {
      if (my_rank % 2 == 0) {

      MPI_Status status;

      if (my_rank != mpi_size - 1) {  // sending right most boundary to next rank
        MPI_Send(&(*first_index_pointer)[(stripe * (my_rank + 1)) - 1][0], size, MPI_DOUBLE, my_rank + 1, 0,
                 MPI_COMM_WORLD);
      }

      if (my_rank != 0) {
        // Sending left most boundary to previous rank
        MPI_Send(&(*first_index_pointer)[(stripe * (my_rank))][0], size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);

        // Receiving left most boundary - 1 from previous rank
        MPI_Recv(&(*first_index_pointer)[(stripe * my_rank) - 1][0], size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD,
                 &status);
      }

      if (my_rank != mpi_size - 1) {
        // Receiving right most boundary + 1 from next rank
        MPI_Recv(&(*first_index_pointer)[(stripe * (my_rank + 1))][0], size, MPI_DOUBLE, my_rank + 1, 0,
                 MPI_COMM_WORLD,
                 &status);
      }
      
    } else {

      MPI_Status status;
      // Recieving the left most boundary from the previous rank
      MPI_Recv(&(*first_index_pointer)[(stripe * my_rank) - 1][0], size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD,
               &status);

      if (my_rank != mpi_size - 1) {
        // Recieving the right most boundary from the next rank when it is not the last rank.
        MPI_Recv(&(*first_index_pointer)[stripe * (my_rank + 1)][0], size, MPI_DOUBLE, my_rank + 1, 0,
                 MPI_COMM_WORLD,
                 &status);

        //Sending the rightmost boundary to the next rank if it is not the last rank.
        MPI_Send(&(*first_index_pointer)[(stripe * (my_rank + 1)) - 1][0], size, MPI_DOUBLE, my_rank + 1, 0,
                 MPI_COMM_WORLD);
      }
//Sending the leftmost boundary to the previous rank.
      MPI_Send(&(*first_index_pointer)[stripe * my_rank][0], size, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);

    }
    }

    // simulate wave diffusion from time = 2
    for (int i = stripe * my_rank; i < (stripe * (my_rank + 1)); i++) {
      for (int j = 0; j < size; j++) {

        if (i == 0 || i == size - 1 || j == 0 || j == size - 1) {
          (*second_index_pointer)[i][j] = 0.0;
        } else {

          (*second_index_pointer)[i][j] = 2.0 * (*first_index_pointer)[i][j] - (*zeroth_index_pointer)[i][j]
              + pow_2_calculation
                  * ((*first_index_pointer)[i + 1][j] + (*first_index_pointer)[i - 1][j]
                      + (*first_index_pointer)[i][j + 1]
                      + (*first_index_pointer)[i][j - 1]
                      - 4.0 * (*first_index_pointer)[i][j]);
        }
      }
    }

    // Exchange of stripe information happening here.
    // Exchange only happens if we need to print i.e if interval > 0.
    if (my_rank == 0 && interval > 0 && t % interval == 0) {

      if (interval > 0 && t % interval == 0) {
        cout << t << endl;
      }

      // Master is receiving stripe information from all the slaves.
      for (int rank = 1; rank < mpi_size; rank++) {
        MPI_Status status;
        MPI_Recv(&(*second_index_pointer)[stripe * rank][0], stripe * size, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD,
                 &status);
      }

      // Printing the formula output to stdout.
      for (int i = 0; i < size; i++) {

        for (int j = 0; j < size; j++) {
          cout << (*second_index_pointer)[i][j] << " ";
        }
        cout << endl;
      }
      cout << endl;

    } else if (interval > 0 && t % interval == 0) {
      // slaves sending the assigned stripe information to the Master.
      MPI_Send(&(*second_index_pointer)[stripe * my_rank][0], stripe * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Pointers rotation logic
    /**
     *  time ->   0 1 2 3 4 5 ....
     *  index ->  0 1 2 0 1 2 ....
     */
    zeroth_index = (zeroth_index + 1) % 3;
    first_index = (first_index + 1) % 3;
    second_index = (second_index + 1) % 3;

    zeroth_index_pointer = &z[zeroth_index];
    first_index_pointer = &z[first_index];
    second_index_pointer = &z[second_index];
  }

  /**
   *  Printing the last time interval value.
   */
  if (my_rank == 0 && interval > 0) {

    cout << max_time - 1 << endl;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        cout << z[2][j][i] << " ";
      }
      cout << endl;
    }
  }
  
  // finish the timer
  if (my_rank == 0) {
    cerr << "Elapsed time = " << time.lap() << endl;
  }

  MPI_Finalize();
  return 0;
}






