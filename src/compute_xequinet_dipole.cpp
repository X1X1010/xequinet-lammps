#include "compute_xequinet_dipole.h"
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

// compute COMPUTE_ID GROUP xequinet/dipole model.jit element1 element2 ...
template<Precision precision> ComputeXequiNetDipole<precision>::ComputeXequiNetDipole(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
  std::cout << "Computing dipole using XequiNet model via LAMMPS interface." << std::endl;
  std::cout << "The model is using precision: " << typeid(data_type).name() << std::endl;

  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  } else {
    device = torch::kCPU;
  }
  std::cout << "The model is on device: " << device << std::endl;

  int ntypes = atom->ntypes;
  if (narg != 4 + ntypes)
    error->all(FLERR,
               "Incorrect args for compute xequinet/dipole, should be "
               "COMPUTE_ID GROUP xequinet/dipole <model>.jit <element1> <element2> ...");
  if (igroup)
    error->all(FLERR, "Compute xequinet/dipole must use group all");

  
  model_path = arg[3];

  // clear elements
  elements.clear();
  for (int i = 0; i < ntypes; ++i) { elements.emplace_back(arg[i + 4]); }

  scalar_flag = 0;
  vector_flag = 1;
  size_vector = 3;
  extscalar = 0;
  extvector = 1;

  vector = new double[size_vector];
  vector[0] = vector[1] = vector[2] = 0.0;
}


template<Precision precision> ComputeXequiNetDipole<precision>::~ComputeXequiNetDipole()
{
  delete[] vector;
}


template<Precision precision> void ComputeXequiNetDipole<precision>::init()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Compute xequinet/dipole requires atom IDs");

  // Request a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);

  // Load the model
  std::unordered_map<std::string, std::string> metadata = {
    {CUTOFF_RADIUS, ""}, {JIT_FUSION_STRATEGY, ""}, {N_SPECIES, ""},
    {PERIODIC_TABLE, ""},
  };
  model = torch::jit::load(std::string(model_path), device, metadata);
  torch::set_num_threads(1);  // single thread for potential nan problem
  model.eval();

  // if the model has not been frozen yet, freeze it
  if (model.hasattr("training")) {
    std::cout << "Freezing the script model." << std::endl;
    model = torch::jit::freeze(model);
  }

  // set fusion strategy
  torch::jit::FusionStrategy strategy;
  strategy = { {torch::jit::FusionBehavior::DYNAMIC, 10} };

  std::stringstream strat_stream(metadata[JIT_FUSION_STRATEGY]);
  std::string fusion_type, fusion_depth;
  while (std::getline(strat_stream, fusion_type, ',')) {
    std::getline(strat_stream, fusion_depth, ';');
    strategy.push_back(
      {fusion_type == "DYNAMIC" ? torch::jit::FusionBehavior::DYNAMIC
                                : torch::jit::FusionBehavior::STATIC,
       std::stoi(fusion_depth)}
    );
  }
  torch::jit::setFusionStrategy(strategy);

  // cutoff radius
  cutoff = std::stod(metadata[CUTOFF_RADIUS]);
  std::cout << "Cutoff radius: " << cutoff << std::endl;
  cutoffsq = cutoff * cutoff;

  // type mapper
  int ntypes = atom->ntypes;
  assert (ntypes == elements.size());
  type_mapper.resize(ntypes, -1);
  std::stringstream ss;
  int n_species = std::stoi(metadata[N_SPECIES]);
  ss << metadata[PERIODIC_TABLE];
  std::cout << "XequiNet type mapping:" << std::endl;
  std::cout << "XequiNet type | XequiNet name | LAMMPS type | LAMMPS name" << std::endl;
  for (int i = 0; i < n_species; ++i) {
    std::string ele;
    ss >> ele;
    for (int itype = 1; itype <= ntypes; ++itype) {
      if (ele.compare(elements[itype - 1]) == 0) {
        type_mapper[itype - 1] = i;
        std::cout << i << " | " << ele << " | " << itype << " | " << elements[itype - 1] << std::endl;
      }
      else if (ele.compare("NULL") == 0) {
        type_mapper[itype - 1] = -1;
      }
    }
  }
}


template<Precision precision> void ComputeXequiNetDipole<precision>::init_list(int /*which*/, NeighList *ptr)
{
  list = ptr;
}


template<Precision precision> void ComputeXequiNetDipole<precision>::compute_vector()
{
  // check if using the periodic boundary conditions
  if (lmp->domain->periodicity[0] || lmp->domain->periodicity[1] || lmp->domain->periodicity[2]) {
    compute_vector_pbc();
  } else {
    compute_vector_non_pbc();
  }
}

template<Precision precision> void ComputeXequiNetDipole<precision>::compute_vector_pbc()
{
  // atom list to point to the same tag, use local index
  int *tag = atom->tag;
  // mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // number of local/real atoms
  int inum = list->inum;
  // number of local/real atoms
  int nlocal = atom->nlocal;
  // atom positions, including ghost atoms
  double **x = atom->x;
  // atom forces
  double **f = atom->f;
  // atom types
  int *type = atom->type;

  // assert (inum == nlocal);

  // number of neighbors per atom
  int *numneigh = list->numneigh;
  // neighbor list per atom
  int **firstneigh = list->firstneigh;

  // assemble pytorch input positions and tags
  torch::Tensor pos_tensor =
      torch::zeros({inum, 3}, torch::TensorOptions().dtype(scalar_type));
  torch::Tensor mapped_type_tensor =
      torch::zeros({inum}, torch::TensorOptions().dtype(torch::kInt64));
  auto pos = pos_tensor.accessor<data_type, 2>();
  auto mapped_type = mapped_type_tensor.accessor<long, 1>();

#pragma omp parallel for
  for (int i = 0; i < inum; ++i) {
    mapped_type[i] = type_mapper[type[i] - 1];
    pos[i][0] = x[i][0];
    pos[i][1] = x[i][1];
    pos[i][2] = x[i][2];
  }

  // number of bonds per atom
  std::vector<int> neigh_per_atom(inum, 0);
#pragma omp parallel for
  for (int ii = 0; ii < inum; ++ii) {
    int i = ilist[ii];
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq < cutoffsq) { neigh_per_atom[ii]++; }
    }
  }

  // cumulative sum of neighbors, for indexing
  std::vector<int> cumsum_neigh_per_atom(inum + 1, 0);
  for (int ii = 1; ii <= inum; ++ii) {
    cumsum_neigh_per_atom[ii] = cumsum_neigh_per_atom[ii - 1] + neigh_per_atom[ii - 1];
  }
  // total number of bonds (sum of all neighbors)
  int nedges = cumsum_neigh_per_atom[inum];

  torch::Tensor edges_tensor =
      torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64));
  auto edges = edges_tensor.accessor<long, 2>();

  torch::Tensor cell_tensor;
  torch::Tensor cell_offsets_tensor;

  cell_tensor = torch::zeros({3, 3}, torch::TensorOptions().dtype(scalar_type));
  auto cell = cell_tensor.accessor<data_type, 2>();
  cell[0][0] = domain->boxhi[0] - domain->boxlo[0];
  cell[1][0] = domain->xy;
  cell[1][1] = domain->boxhi[1] - domain->boxlo[1];
  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->boxhi[2] - domain->boxlo[2];

  torch::Tensor pbc_tensor;
  pbc_tensor = torch::zeros({3}, torch::TensorOptions().dtype(torch::kBool));
  auto pbc = pbc_tensor.accessor<bool, 1>();
  pbc[0] = domain->periodicity[0];
  pbc[1] = domain->periodicity[1];
  pbc[2] = domain->periodicity[2];

  cell_offsets_tensor = torch::zeros({nedges, 3}, torch::TensorOptions().dtype(scalar_type));
  auto cell_offsets = cell_offsets_tensor.accessor<data_type, 2>();
  double *h_inv = domain->h_inv;

  // calculate cell shifts and cell offsets
  double s0, s1, s2, cs0, cs1, cs2;
#pragma omp parallel for
  for (int ii = 0; ii < inum; ++ii) {
    int i = ilist[ii];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    
    int edge_counter = cumsum_neigh_per_atom[ii];
    for (int jj = 0; jj < jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      int j_real = atom->map(tag[j]);
      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq < cutoffsq) {
        edges[CENTER_IDX][edge_counter] = i;
        edges[NEIGHBOR_IDX][edge_counter] = j_real;

        if (j == j_real) {
          cell_offsets[edge_counter][0] = 0.0;
          cell_offsets[edge_counter][1] = 0.0;
          cell_offsets[edge_counter][2] = 0.0;
        } else {
          // calculate cell offsets
          s0 = x[j][0] - x[j_real][0];
          s1 = x[j][1] - x[j_real][1];
          s2 = x[j][2] - x[j_real][2];
          cs0 = s0 * h_inv[0] + s1 * h_inv[5] + s2 * h_inv[4];
          cs1 = s1 * h_inv[1] + s2 * h_inv[3];
          cs2 = s2 * h_inv[2];

          cell_offsets[edge_counter][0] = std::round(cs0);
          cell_offsets[edge_counter][1] = std::round(cs1);
          cell_offsets[edge_counter][2] = std::round(cs2);
        }
        ++edge_counter;
      }
    }
  }

  // prepare input
  c10::Dict<std::string, torch::Tensor> input;
  input.insert(ATOMIC_NUMBERS, mapped_type_tensor.to(device));
  input.insert(POSITIONS, pos_tensor.to(device));
  input.insert(CELL_OFFSETS, cell_offsets_tensor.to(device));
  input.insert(CELL, cell_tensor.unsqueeze(0).to(device));
  input.insert(PBC, pbc_tensor.unsqueeze(0).to(device));
  input.insert(EDGE_INDEX, edges_tensor.to(device));
  std::vector<torch::jit::IValue> input_vector(1, input);

  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor dipole_tensor = output.at(DIPOLE).toTensor().cpu().reshape({3});
  auto dipole = dipole_tensor.accessor<data_type, 1>();
  for (int i = 0; i < 3; ++i) {
    vector[i] = dipole[i];
  }
}

template<Precision precision> void ComputeXequiNetDipole<precision>::compute_vector_non_pbc()
{
  // mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // number of local/real atoms
  int inum = list->inum;
  // number of local/real atoms
  int nlocal = atom->nlocal;
  // atom positions, including ghost atoms
  double **x = atom->x;
  // atom forces
  double **f = atom->f;
  // atom types
  int *type = atom->type;

  // assert (inum == nlocal);

  // number of ghost atoms
  int nghost = atom->nghost;
  // total number of atoms
  int ntotal = inum + nghost;

  // number of neighbors per atom
  int *numneigh = list->numneigh;
  // neighbor list per atom
  int **firstneigh = list->firstneigh;

  // total number of bonds (sum of all neighbors)
  int nedges = 0;

  // number of bonds per atom
  std::vector<int> neigh_per_atom(inum, 0);

  // count total number of bonds
#pragma omp parallel for reduction(+ : nedges)
  for (int ii = 0; ii < inum; ++ii) {
    int i = ilist[ii];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx * dx + dy * dy + dz * dz;
      if (rsq < cutoffsq) {
        neigh_per_atom[ii]++;
        nedges++;
      }
    }
  }

  // cumlative sum of neighbors, for indexing
  std::vector<int> cumsum_neigh_per_atom(inum);
#pragma omp parallel for
  for (int ii = 1; ii < inum; ++ii) {
    cumsum_neigh_per_atom[ii] = cumsum_neigh_per_atom[ii - 1] + neigh_per_atom[ii - 1];
  }

  torch::Tensor pos_tensor =
      torch::zeros({ntotal, 3}, torch::TensorOptions().dtype(scalar_type));
  torch::Tensor mapped_type_tensor =
      torch::zeros({ntotal}, torch::TensorOptions().dtype(torch::kInt64));
  auto pos = pos_tensor.accessor<data_type, 2>();
  auto mapped_type = mapped_type_tensor.accessor<long, 1>();

  torch::Tensor edges_tensor =
      torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64));
  auto edges = edges_tensor.accessor<long, 2>();

  // loop over atoms and neighbors to fill in pos and edges tensors
#pragma omp parallel for
  for (int ii = 0; ii < ntotal; ++ii) {
    int i = ilist[ii];
    mapped_type[ii] = type_mapper[type[i] - 1];

    pos[i][0] = x[i][0];
    pos[i][1] = x[i][1];
    pos[i][2] = x[i][2];

    if (ii < inum) {
      int jnum = numneigh[i];
      int *jlist = firstneigh[i];

      int edge_counter = cumsum_neigh_per_atom[ii];
      for (int jj = 0; jj < jnum; ++jj) {
        int j = jlist[jj];
        j &= NEIGHMASK;

        double dx = x[i][0] - x[j][0];
        double dy = x[i][1] - x[j][1];
        double dz = x[i][2] - x[j][2];

        double rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < cutoffsq) {
          edges[CENTER_IDX][edge_counter] = i;
          edges[NEIGHBOR_IDX][edge_counter] = j;
          ++edge_counter;
        }
      }
    }
  }

  c10::Dict<std::string, torch::Tensor> input;
  input.insert(ATOMIC_NUMBERS, mapped_type_tensor.to(device));
  input.insert(POSITIONS, pos_tensor.to(device));
  input.insert(EDGE_INDEX, edges_tensor.to(device));
  std::vector<torch::jit::IValue> input_vector(1, input);
  
  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor dipole_tensor = output.at(DIPOLE).toTensor().cpu().reshape({3});
  auto dipole = dipole_tensor.accessor<data_type, 1>();
  for (int i = 0; i < 3; ++i) {
    vector[i] = dipole[i];
  }
}

namespace LAMMPS_NS {
  template class ComputeXequiNetDipole<low>;
  template class ComputeXequiNetDipole<high>;
}  // namespace LAMMPS_NS