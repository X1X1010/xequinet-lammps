#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(xequinet/dipole,ComputeXequiNetDipole<low>);
ComputeStyle(xequinet32/dipole,ComputeXequiNetDipole<low>);
ComputeStyle(xequinet64/dipole,ComputeXequiNetDipole<high>);
// clang-format on
#else

#ifndef LMP_COMPUTE_XEQUINET_DIPOLE_H
#define LMP_COMPUTE_XEQUINET_DIPOLE_H

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
#define DIPOLE "dipole"

#include <torch/torch.h>
#include <string>
#include <vector>

#include "compute.h"
#include "precision.h"

namespace LAMMPS_NS {

template<Precision precision> class ComputeXequiNetDipole : public Compute {
  public:
    ComputeXequiNetDipole(class LAMMPS *, int, char **);
    virtual ~ComputeXequiNetDipole();
    virtual void init();
    virtual void init_list(int, class NeighList *);
    virtual void compute_vector();
    virtual void compute_vector_pbc();
    virtual void compute_vector_non_pbc();
  protected:
    double cutoff;
    double cutoffsq;
    class NeighList *list;
    std::string model_path;
    std::vector<std::string> elements;
    torch::jit::Module model;
    torch::Device device = torch::kCPU;
    std::vector<int> type_mapper;

    typedef typename std::conditional_t<precision == low, float, double> data_type;

    torch::ScalarType scalar_type = torch::CppTypeToScalarType<data_type>();
};
}  // namespace LAMMPS_NS

#endif
#endif