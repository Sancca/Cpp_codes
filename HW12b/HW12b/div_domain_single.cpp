#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <math.h>
#include <vector>
#include <thread>
#include <mpi.h>
using namespace std;

void outputField(double** data, int ni, int nj, int mpi_rank);

// support for complex numbers
struct Complex {
	double r;
	double i;

	Complex(double a, double b) : r(a), i(b) { }

	double magnitude2() { return r * r + i * i; }

	Complex operator*(const Complex& a) {
		return Complex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	Complex operator+(const Complex& a) {
		return Complex(r + a.r, i + a.i);
	}
};

// computes value of julia set at [i,j]*/
double juliaValue(int i, int j, int ni, int nj)
{
	double fi = -1.0 + 2.0 * i / ni;	// fi = [-1:1)
	double fj = -1.0 + 2.0 * j / nj;	// fj = [-1:1)

	Complex c(-0.8, 0.156);	// coefficient for the image
	Complex a(fi, fj);		// pixel pos as a complex number

	int k;
	for (k = 0; k < 200; k++) {
		a = a * a + c;
		if (a.magnitude2() > 1000) break;	// check for divergence
	}
	return k;				// return 
}

void calculatePixels(int i_start, int i_end, int ni, int nj, double** julia) {
	for (int i = i_start; i < i_end; i++)
		for (int j = 0; j < nj; j++)
			julia[i][j] = juliaValue(i, j, ni, nj);
}

int main(int num_args, char** args)
{
	MPI_Init(&num_args, &args);  // initialize MPI

	// figure out what rank I am and the total number of processes
	int mpi_rank, mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	const int ni = 4000;
	const int nj = 4000;

	/*allocate memory for our domain*/
	double** julia = new double* [ni];
	for (int i = 0; i < ni; i++) julia[i] = new double[nj];

	//Initialize array with zeros
	for (int i = 0; i < ni; i++) {
		for (int j = 0; j < nj; j++) {
			julia[i][j] = 0;
		}
	}
	
	// Domain Decomposition (utilize whole matrix, but perform computation in blocks for each array)
	int block_div = ni / mpi_size;
	int my_block_start_i = mpi_rank * block_div;
	int my_block_end_i = (mpi_rank + 1) * block_div;
	
	// final rank correction to index
	if (mpi_rank == mpi_size - 1) {
		my_block_end_i = ni - block_div * (mpi_size - 1);
	}

	// Each process performs calculation individually
	calculatePixels(my_block_start_i, my_block_end_i, ni, nj, julia);

	// Send calculations to root 
	if (mpi_rank > 0) {
		for (int i = my_block_start_i; i < my_block_end_i; i++) {
			MPI_Send(julia[i], nj, MPI_DOUBLE, 0, 21, MPI_COMM_WORLD);
		}
	}

	double* julia_temp = new double[nj]; //Allocates memory

	// Placing data in single 2D array
	if (mpi_rank == 0) {
		for (int r = 1; r < mpi_size; r++) {
			MPI_Status status;
			// Calculate index (same as above)
			int index_start = r * block_div;
			int index_end = (r + 1) * block_div;
			if (r == mpi_size - 1) {
				index_end = ni - block_div * (mpi_size - 1);
			}

			// Recieve array data from nodes
			for (int i = index_start; i < index_end; i++) {
				MPI_Recv(&julia_temp, nj, MPI_DOUBLE, r, 21, MPI_COMM_WORLD, &status);
			}

			// Update root julia 
			for (int i = index_start; i < index_end; i++) {
				for (int j = 0; j < nj; j++)
					julia[i][j] = julia_temp[j];
			}

		}
	}
	// Write file
	outputField(julia, ni, nj, 0);

	// free memory
	for (int i = 0; i < ni; i++) delete[] julia[i];
	delete[] julia;
	delete[] julia_temp;

	MPI_Finalize(); // End MPI session
	return 0;
}

/*saves output in VTK format*/
void outputField(double** data, int ni, int nj, int mpi_rank)
{
	stringstream name;
	name << "julia" << mpi_rank << ".vti";

	/*open output file*/
	ofstream out(name.str());
	if (!out.is_open()) { cerr << "Could not open " << name.str() << endl; return; }

	/*ImageData is vtk format for structured Cartesian meshes*/
	out << "<VTKFile type=\"ImageData\">\n";
	out << "<ImageData Origin=\"" << "0 0 0\" ";
	out << "Spacing=\"1 1 1\" ";
	out << "WholeExtent=\"0 " << ni - 1 << " 0 " << nj - 1 << " 0 0\">\n";

	/*output data stored on nodes (point data)*/
	out << "<PointData>\n";

	/*potential, scalar*/
	out << "<DataArray Name=\"julia\" NumberOfComponents=\"1\" format=\"ascii\" type=\"Float32\">\n";
	for (int j = 0; j < nj; j++)
	{
		for (int i = 0; i < ni; i++) out << data[i][j] << " ";
		out << "\n";
	}
	out << "</DataArray>\n";

	/*close out tags*/
	out << "</PointData>\n";
	out << "</ImageData>\n";
	out << "</VTKFile>\n";
	out.close();
}
