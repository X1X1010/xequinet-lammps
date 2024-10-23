/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------
   Contributing authors: Wenjie Yan, Yicheng Chen (The XDFT group at Fudan)
---------------------------------------------------------------------------- */

#include "pair_xequinet.h"
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

template<Precision precision> PairXequiNet<precision>::PairXequiNet(LAMMPS *lmp) : Pair(lmp) {

  restartinfo = 0;
  manybody_flag = 1;

  std::cout << "Running XequiNet model via LAMMPS interface." << std::endl;
  std::cout << "The model is using precision: " << typeid(data_type).name() << std::endl;

  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  } else {
    device = torch::kCPU;
  }
  std::cout << "The model is on device: " << device << std::endl;
}

template<Precision precision> PairXequiNet<precision>::~PairXequiNet() {

  if (allocated) {
      memory->destroy(setflag);
      memory->destroy(cutsq);
  }
}

template<Precision precision> void PairXequiNet<precision>::init_style() {

  if (atom->tag_enable == 0) error->all(FLERR, "Pair style XequiNet requires atom IDs");

  // Request a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  
  if (force->newton_pair == 0) error->all(FLERR, "Pair style XequiNet requires newton pair on");
}

template<Precision precision> double PairXequiNet<precision>::init_one(int i, int j) {
    return cutoff;
}

template <Precision precision> void PairXequiNet<precision>::allocate() {

  int n = atom->ntypes;

  // atom types are 1-based
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  allocated = 1;
}

template <Precision precision> void PairXequiNet<precision>::settings(int narg, char ** /*arg*/) {
  // "xequinet" must be the first argument after pair_style
  if (narg > 0) error->all(FLERR, "Illegal pair_style command for XequiNet, too many arguments.");
}

template <Precision precision> void PairXequiNet<precision>::coeff(int narg, char** arg) {
  // parse coefficients of pair_coeff command. e.g.
  // pair_coeff xequinet * * model.jit O H
  if (!allocated) allocate();

  int ntypes = atom->ntypes;

  // should be exactly 3 arguments following "pair_coeff"
  if (narg != 3 + ntypes)
    error->all(FLERR,
               "Incorrect args for pair coefficients, should * * "
                "<model>.jit <element1> <element2> ...");
  // ensure args are "* *"
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Illegal pair_coeff command for XequiNet");

  for (int i = i; i <= ntypes; ++i)
    for (int j = i; j <= ntypes; ++j) setflag[i][j] = 0;

  std::vector<std::string> elements(ntypes);
  for (int i = 0; i < ntypes; ++i) { elements[i] = arg[i + 3]; }

  std::unordered_map<std::string, std::string> metadata = {
    {CUTOFF_RADIUS, ""}, {JIT_FUSION_STRATEGY, ""}, {N_SPECIES, ""},
    {PERIODIC_TABLE, ""},
  };

  model = torch::jit::load(std::string(arg[2]), device, metadata);

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
    }
  }
  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; ++i) {
    for (int j = i; j <= ntypes; ++j) {
      if ((type_mapper[i - 1] >= 0) && (type_mapper[j - 1] >= 0)) { setflag[i][j] = 1; }
    }
  }
}

template<Precision precision> void PairXequiNet<precision>::compute(int eflag, int vflag) {

  // check flag
  ev_init(eflag, vflag);
  if (vflag_atom) { error->all(FLERR, "Pair style XequiNet does not support atom virials"); }

  // check if using the periodic boundary conditions
  if (lmp->domain->periodicity[0] || lmp->domain->periodicity[1] || lmp->domain->periodicity[2]) {
    compute_pbc(eflag, vflag);
  } else {
    compute_non_pbc(eflag, vflag);
  }
}


template<Precision precision> void PairXequiNet<precision>::compute_pbc(int eflag, int vflag) {

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

  assert(inum == nlocal);

  // number of neighbors per atom
  int *numneigh = list->numneigh;
  // neighbor list per atom
  int **firstneigh = list->firstneigh;

  // assemble pytorch input positions ang tags
  torch::Tensor pos_tensor =
      torch::zeros({nlocal, 3}, torch::TensorOptions().dtype(scalar_type));
  torch::Tensor mapped_type_tensor =
      torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  auto pos = pos_tensor.accessor<data_type, 2>();
  auto mapped_type = mapped_type_tensor.accessor<long, 1>();

#pragma omp parallel for
  for (int i = 0; i < nlocal; ++i) {
    mapped_type[i] = type_mapper[type[i] - 1];
    pos[i][0] = x[i][0];
    pos[i][1] = x[i][1];
    pos[i][2] = x[i][2];
  }

  // number of bonds per atom
  std::vector<int> neigh_per_atom(nlocal, 0);
#pragma omp parallel for
  for (int ii = 0; ii < nlocal; ++ii) {
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
      if (rsq <= cutoffsq) { neigh_per_atom[ii]++; }
    }
  }

  // cumulative sum of neighbors, for indexing
  std::vector<int> cumsum_neigh_per_atom(nlocal + 1, 0);
  for (int ii = 1; ii <= nlocal; ++ii) {
    cumsum_neigh_per_atom[ii] = cumsum_neigh_per_atom[ii - 1] + neigh_per_atom[ii - 1];
  }
  // total number of bonds (sum of all neighbors)
  int nedges = cumsum_neigh_per_atom[nlocal];

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
  for (int ii = 0; ii < nlocal; ++ii) {
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
      if (rsq <= cutoffsq) {
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
  const bool fflag_input = true;
  const bool vflag_input = (vflag > 0);
  std::unordered_map<std::string, torch::IValue> kwargs;
  kwargs["compute_forces"] = fflag_input;
  kwargs["compute_virial"] = vflag_input;

  auto output = model.forward(input_vector, kwargs).toGenericDict();

  torch::Tensor forces_tensor = output.at(FORCES).toTensor().cpu();
  auto forces = forces_tensor.accessor<data_type, 2>();
  torch::Tensor atomic_energies_tensor = output.at(ATOMIC_ENERGIES).toTensor().cpu();
  auto atomic_energies = atomic_energies_tensor.accessor<data_type, 1>();
  torch::Tensor energy = output.at(TOTAL_ENERGY).toTensor().cpu();

  // write forces and atomic energies back to LAMMPS
  eng_vdwl = energy.item<double>();
#pragma omp parallel for
  for (int i = 0; i < nlocal; ++i) {
    f[i][0] += forces[i][0];
    f[i][1] += forces[i][1];
    f[i][2] += forces[i][2];

    if (eflag_atom) eatom[i] = atomic_energies[i];
  }
  
  if (vflag) {
    torch::Tensor v_tensor = output.at(VIRIAL).toTensor().cpu();
    auto v = v_tensor.accessor<data_type, 3>();
    // convert 3x3 virial tensor to 6 virial components
    // first dimension of virial is batch
    virial[0] = v[0][0][0];
    virial[1] = v[0][1][1];
    virial[2] = v[0][2][2];
    virial[3] = v[0][0][1];
    virial[4] = v[0][0][2];
    virial[5] = v[0][1][2];
  }
}


template<Precision precision> void PairXequiNet<precision>::compute_non_pbc(int eflag, int vflag) {

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

  assert(inum == nlocal);

  // number of ghost atoms
  int nghost = atom->nghost;
  // total number of atoms
  int ntotal = nlocal + nghost;

  // number of neighbors per atom
  int *numneigh = list->numneigh;
  // neighbor list per atom
  int **firstneigh = list->firstneigh;

  // total number of bonds (sum of all neighbors)
  int nedges = 0;

  // number of bonds per atom
  std::vector<int> neigh_per_atom(nlocal, 0);

  // count total number of bonds
#pragma omp parallel for reduction(+ : nedges)
  for (int ii = 0; ii < nlocal; ++ii) {
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
      if (rsq <= cutoffsq) {
        neigh_per_atom[ii]++;
        nedges++;
      }
    }
  }

  // cumulative sum of neighbors, for indexing
  std::vector<int> cumsum_neigh_per_atom(nlocal);
#pragma omp parallel for
  for (int ii = 1; ii < nlocal; ++ii) {
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

    if (ii < nlocal) {
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
        if (rsq <= cutoffsq) {
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
  const bool fflag_input = true;
  const bool vflag_input = (vflag > 0);
  std::unordered_map<std::string, torch::IValue> kwargs;
  kwargs["compute_forces"] = fflag_input;
  kwargs["compute_virial"] = vflag_input;

  auto output = model.forward(input_vector, kwargs).toGenericDict();

  torch::Tensor forces_tensor = output.at(FORCES).toTensor().cpu();
  auto forces = forces_tensor.accessor<data_type, 2>();
  torch::Tensor atomic_energies_tensor = output.at(ATOMIC_ENERGIES).toTensor().cpu();
  auto atomic_energies = atomic_energies_tensor.accessor<data_type, 1>();
  torch::Tensor energy = output.at(TOTAL_ENERGY).toTensor().cpu();

  eng_vdwl = energy.item<double>();
  // write forces and atomic energies back to LAMMPS
#pragma omp parallel for
  for (int ii = 0; ii < ntotal; ++ii) {
    int i = ilist[ii];
    f[i][0] += forces[i][0];
    f[i][1] += forces[i][1];
    f[i][2] += forces[i][2];

    if (eflag_atom && ii < inum) eatom[i] = atomic_energies[i];
  }
  
  if (vflag_fdotr) virial_fdotr_compute();
}

namespace LAMMPS_NS {
  template class PairXequiNet<low>;
  template class PairXequiNet<high>;
}  // namespace LAMMPS_NS