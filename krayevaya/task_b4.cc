#include <assert.h>
#include <cstddef>
#include <ctype.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


#define DOTS 3000


double h = 0;
// h * M = xM - x0;
int M = 0;

constexpr double A = 100000.0;

constexpr double x0 = -10;
constexpr double xM = 10;

constexpr double a = sqrt(2);
constexpr double b = a;

// y'' = f(y) = y^3 - y)
double f(double y) { return -A * (y * y * y - y); };

double deriv_f(double y) { return -A * (3 * y * y - 1); }
//_________________________

// y(x) = y0(x) + v(x), y0 - rough linear solution
double y0(double x) { return a + (b - a) / (xM - x0) * (x - x0); }

double f0(double x) { return f(y0(x)); }

double deriv_f0(double x) { return deriv_f(y0(x)); }

// a * v_{m-1} + b * v_m + c * v_{m+1} = g
typedef struct ReductionCoefs {
  double a;
  double b;
  double c;
  double g;
  // Run one reduction iteration
  void fillInitCoeffs(double x) {
    a = 1 - deriv_f0(x - h) * h * h / 12;
    b = -2 - 5 * deriv_f0(x) * h * h / 6;
    c = 1 - deriv_f0(x + h) * h * h / 12;
    g = (f0(x + h) / 12 + 5 * f0(x) / 6 + f0(x - h) / 12) * h * h;
  };
  void fillNewCoeffs(struct ReductionCoefs Prev, struct ReductionCoefs Curr,
                     struct ReductionCoefs Next) {
    a = -Prev.a * Curr.a / Prev.b;
    b = (Curr.b * Prev.b * Next.b - Curr.a * Prev.c * Next.b -
         Prev.b * Curr.c * Next.a) /
        (Next.b * Prev.b);
    c = -Curr.c * Next.c / Next.b;
    g = (Curr.g * Prev.b * Next.b - Curr.a * Prev.g * Next.b -
         Curr.c * Prev.b * Next.g) /
        (Next.b * Prev.b);
  };
} ReductionCoefs;

MPI_Datatype mpi_coeffs_type;

static int getPositivePowerOfTwo(int n) {
  if (n == 0)
    return -1;

  if (ceil(log2(n)) != floor(log2(n)))
    return -1;
  return log2(n);
}

void calcBoundaryTask(int rank, int Size);

int main(int Argc, char **Argv) {
  int Size, rank;

  // h must be << 1 / sqrt(A);
  // h * M = xM - x0, h << 1 / sqrt(A); M = 2^p + 1;
  h = 1 / sqrt(A) / DOTS;
  M = (xM - x0) / h;

  --M;
  for (unsigned k = 0; k <= 4; ++k)
    M |= M >> (1 << k);
  ++M;
  h = (xM - x0) / M;

  MPI_Init(&Argc, &Argv);
  MPI_Comm_size(MPI_COMM_WORLD, &Size);
  assert(Size <= 32);
  assert(getPositivePowerOfTwo(Size) >= 0);

  /* Create a type for struct ReductionCoefs */
  const int nitems = 4;
  int blocklengths[4] = {1, 1, 1, 1};
  MPI_Datatype types[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
  MPI_Aint offsets[4];

  offsets[0] = offsetof(ReductionCoefs, a);
  offsets[1] = offsetof(ReductionCoefs, b);
  offsets[2] = offsetof(ReductionCoefs, c);
  offsets[3] = offsetof(ReductionCoefs, g);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                         &mpi_coeffs_type);
  MPI_Type_commit(&mpi_coeffs_type);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  static double start = MPI_Wtime();

  calcBoundaryTask(rank, Size);

  double FullTime = MPI_Wtime() - start;

  if (rank == 0)
    printf("Time: %f sec.\n", FullTime);
  MPI_Type_free(&mpi_coeffs_type);
  MPI_Finalize();
  return 0;
}

void getStartAndLenX(int *StartX, int *LenX, int rank, int Size) {
  int SegmentForOneProc = M / Size;
  int Start = SegmentForOneProc * rank;
  int End = SegmentForOneProc * (rank + 1);
  assert(M % Size == 0);
  *LenX = End - Start;
  *StartX = Start;
}

void dumpSolution(int StartX, int EndX, double *Solutions) {
  const int Divisor = 4096 * 2 * log(A / 100);
  for (int J = StartX; J < EndX; J += Divisor)
    printf("%lf\t%lf\n", x0 + J * h,
            Solutions[J] + y0(x0 + J * h));
}

/* LINEAR APPROXIMATION:
 * A rough linear approximation, y0(x), is defined as:
 * 
 *     y0(x) = a + (b - a) / (xM - x0) * (x - x0),
 * 
 * which satisfies the boundary conditions.
*/
double solveLinear(ReductionCoefs C, double Prev, double Next) {
  return (C.g - C.a * Prev - C.c * Next) / C.b;
}
/* DISCRETIZATION:
 * The solution domain [x0, xM] is discretized into M points with step size h:
 * 
 *     h = (xM - x0) / M,
 * 
 * where M is chosen to ensure the stability and accuracy of the solution. The
 * discrete version of the equation is solved iteratively using a reduction
 * scheme:
 * 
 *     a * v_{m-1} + b * v_m + c * v_{m+1} = g,
 * 
 * where:
 *   - v_m is the correction to the linear approximation y0(x),
 *   - a, b, c are coefficients derived from the linearization of f(y),
 *   - g is the source term derived from the discretized equation.
 * 
 * REDUCTION PROCESS:
 * The coefficients are updated iteratively to solve the tridiagonal system
 * of equations using parallel processing. Boundary conditions are handled
 * explicitly during the reduction.
 */
void calcBoundaryTask(int rank, int Size) {
  int StartX, LenX;
  getStartAndLenX(&StartX, &LenX, rank, Size);
  int SegmentForOneProc = M / Size;

  double LeftBound, RightBound;
  ReductionCoefs *Coeffs =
      (ReductionCoefs *)calloc(SegmentForOneProc + 1, sizeof(ReductionCoefs));

  int p = log2(M);
  assert(p > 4);
  int ReductionSteps = p - getPositivePowerOfTwo(Size);

  if (rank == 0) {
    ReductionCoefs *CoeffsArray =
        (ReductionCoefs *)calloc(M + 1, sizeof(ReductionCoefs));
    double *Solution = (double *)calloc(M + 1, sizeof(double));
    Solution[0] = 0;
    Solution[M] = 0;

    for (int i = 0; i <= SegmentForOneProc; ++i)
      CoeffsArray[i].fillInitCoeffs(i * h);
    for (int Proc = 1; Proc < Size; ++Proc) {
      MPI_Recv(CoeffsArray + Proc * SegmentForOneProc, SegmentForOneProc + 1,
               mpi_coeffs_type, Proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int H = 1, PrevH = 1, RedStep = 0;
    for (; RedStep < p - 1; ++RedStep) {
      H *= 2;
      int i = 0;
      for (i = H; i < M; i += H) {
        CoeffsArray[i].fillNewCoeffs(CoeffsArray[i - PrevH], CoeffsArray[i],
                                     CoeffsArray[i + PrevH]);
      }
      assert(i == M);
      PrevH = H;
    }

    for (; RedStep >= ReductionSteps; --RedStep) {
      int i = 0;
      for (i = H; i < M; i += PrevH)
        Solution[i] =
            solveLinear(CoeffsArray[i], Solution[i - H], Solution[i + H]);
      assert(H == PrevH || M + H == i);
      PrevH = H;
      H /= 2;
    }
    LeftBound = Solution[0];
    RightBound = Solution[SegmentForOneProc];
    for (int i = 0; i <= SegmentForOneProc; ++i) {
      Coeffs[i] = CoeffsArray[i];
    }
    for (int Proc = 1; Proc < Size; ++Proc) {
      double Bounds[2] = {Solution[PrevH * Proc], Solution[PrevH * (Proc + 1)]};
      MPI_Send(Bounds, 2, MPI_DOUBLE, Proc, 0, MPI_COMM_WORLD);
    }

    free(CoeffsArray);
    free(Solution);
  } else {
    for (int i = 0; i <= SegmentForOneProc; ++i)
      Coeffs[i].fillInitCoeffs((SegmentForOneProc * rank + i) * h);
    MPI_Send(Coeffs, SegmentForOneProc + 1, mpi_coeffs_type, 0, 0, MPI_COMM_WORLD);

    int H = 1, PrevH = 1, RedStep = 0;
    for (; RedStep < ReductionSteps; ++RedStep) {
      H *= 2;
      int i = 0;
      for (i = H; i < SegmentForOneProc; i += H) {
        Coeffs[i].fillNewCoeffs(Coeffs[i - PrevH], Coeffs[i],
                                Coeffs[i + PrevH]);
      }
      PrevH = H;
    }
    double Bounds[2];
    MPI_Recv(&Bounds, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    LeftBound = Bounds[0];
    RightBound = Bounds[1];
  }

  double *Solution = (double *)calloc(SegmentForOneProc + 1, sizeof(double));
  Solution[0] = LeftBound;
  Solution[SegmentForOneProc] = RightBound;

  int PrevH = SegmentForOneProc / 2, H = PrevH;
  for (; H > 0; H /= 2) {
    int i = 0;
    for (i = H; i < SegmentForOneProc; i += PrevH)
      Solution[i] = solveLinear(Coeffs[i], Solution[i - H], Solution[i + H]);
    PrevH = H;
  }

  if (rank == 0) {
    double *Result = (double *)calloc(M + 1, sizeof(double));
    for (int i = 0; i < SegmentForOneProc; ++i)
      Result[i] = Solution[i];
    for (int Proc = 1; Proc < Size; ++Proc) {
      MPI_Recv(Result + Proc * SegmentForOneProc, SegmentForOneProc, MPI_DOUBLE,
               Proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    dumpSolution(0, M + 1, Result);
    free(Result);
  } else {
    MPI_Send(Solution, SegmentForOneProc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  free(Solution);
  free(Coeffs);
}
