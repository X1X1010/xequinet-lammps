// PairXequiNet.h  wjyan 2024/09 

#ifdef PAIR_CLASS

PairStyle(xequinet, PairXequiNet<low>)
PairStyle(xequinet32, PairXequiNet<low>)
PairStyle(xequinet64, PairXequiNet<high>)

#else

#ifndef LMP_PAIR_XEQUINET_H
#define LMP_PAIR_XEQUINET_H

#define CUTOFF_RADIUS "cutoff_radius"
#define JIT_FUSION_STRATEGY "jit_fusion_strategy"
#define N_SPECIES "n_species"
#define PERIODIC_TABLE "periodic_table"
#define POSITIONS "pos"
#define ATOMIC_NUMBERS "atomic_numbers"
#define EDGE_INDEX "edge_index"
#define CELL_OFFSETS "cell_offsets"
#define CELL "cell"
#define PBC "pbc"
#define CENTER_IDX 0
#define NEIGHBOR_IDX 1
#define TOTAL_ENERGY "energy"
#define FORCES "forces"
#define ATOMIC_ENERGIES "atomic_energies"
#define VIRIAL "virial"


#include <torch/torch.h>
#include <vector>

#include "pair.h"
#include "precision.h"

namespace LAMMPS_NS {

template<Precision precision> class PairXequiNet: public Pair {
  public:
    PairXequiNet(class LAMMPS *);
    virtual ~PairXequiNet();
    virtual void compute(int, int);
    virtual void compute_non_pbc(int, int);
    virtual void compute_pbc(int, int);
    void settings(int, char **);
    virtual void coeff(int, char **);
    virtual void init_style();
    virtual double init_one(int, int);
    
  protected:
    virtual void allocate();
    double cutoff; 
    double cutoffsq;
    torch::jit::Module model;
    torch::Device device = torch::kCPU;
    std::vector<int> type_mapper;

    typedef typename std::conditional_t<precision == low, float, double> data_type;

    torch::ScalarType scalar_type = torch::CppTypeToScalarType<data_type>();
};

}  // namespace LAMMPS_NS


#endif 
#endif
