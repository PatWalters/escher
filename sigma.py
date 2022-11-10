import sys
from io import StringIO

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
import itertools
from collections import defaultdict
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem import rdChemReactions
import useful_rdkit_utils as uru

# This code was borrowed from the RDKit Cookbook
def find_bond_groups(mol):
    """Find groups of contiguous rotatable bonds and return them sorted by decreasing size"""
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    rot_bond_set = set([mol.GetBondBetweenAtoms(*ap).GetIdx() for ap in rot_atom_pairs])
    rot_bond_groups = []
    while rot_bond_set:
        i = rot_bond_set.pop()
        connected_bond_set = set([i])
        stack = [i]
        while (stack):
            i = stack.pop()
            b = mol.GetBondWithIdx(i)
            bonds = []
            for a in (b.GetBeginAtom(), b.GetEndAtom()):
                bonds.extend([b.GetIdx() for b in a.GetBonds() if (
                        (b.GetIdx() in rot_bond_set) and (not (b.GetIdx() in connected_bond_set)))])
            connected_bond_set.update(bonds)
            stack.extend(bonds)
        rot_bond_set.difference_update(connected_bond_set)
        rot_bond_groups.append(tuple(connected_bond_set))
    return tuple(sorted(rot_bond_groups, reverse=True, key=lambda x: len(x)))


def num_contiguous_rotatable_bonds(mol):
    # Find groups of contiguous rotatable bonds in mol
    bond_groups = find_bond_groups(mol)
    # As bond groups are sorted by decreasing size, the size of the first group (if any)
    # is the largest number of contiguous rotatable bonds in mol
    return len(bond_groups[0]) if bond_groups else 0


def get_spiro_atoms(mol):
    """
    Get atoms that are part of a spiro fusion
    :param mol: input RDKit molecule
    :return: a list of atom numbers for atims that are the centers of spiro fusions
    """
    info = mol.GetRingInfo()
    ring_sets = [set(x) for x in info.AtomRings()]
    spiro_atoms = []
    for i, j in itertools.combinations(ring_sets, 2):
        i_and_j = (i.intersection(j))
        if len(i_and_j) == 1:
            spiro_atoms += list(i_and_j)
    return spiro_atoms


def max_ring_size(mol):
    """
    Get the size of the largest ring in a molecule
    :param mol: input_molecule
    :return: size of the largest ring or 0 for an acyclic molecule
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if len(atom_rings) == 0:
        return 0
    else:
        return max([len(x) for x in ri.AtomRings()])


def gen_3d(mol, remove_H=True):
    """
    Generate a 3D structure for a RDKit molecule
    :param mol: input molecule
    :param remove_H: remove added hydrogens
    :return: molecule with 3D coordinates, None if embedding failed
    """
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, params=params) == 0:
        AllChem.MMFFOptimizeMolecule(mol)
    else:
        return None
    if remove_H:
        mol = Chem.RemoveHs(mol)
    return mol


def get_centers(mol):
    """
    Get graph centers based on https://pubs.acs.org/doi/abs/10.1021/ci60022a011
    :param mol: input molecule
    :return: list atom indices for centers
    """
    # find atoms with the maximum value for the minimum distance
    dm = Chem.GetDistanceMatrix(mol)
    max_dist = dm.max(axis=0)
    centers = np.argwhere(max_dist == np.amin(max_dist)).flatten()
    # use the distance sum as a tie breaker
    if len(centers) > 1:
        sum_dist = dm.sum(axis=1).astype(int)
        sum_dist = [sum_dist[x] for x in centers]
        min_idx = np.argwhere(sum_dist == min(sum_dist)).flatten()
        centers = centers[min_idx]
    return centers.tolist()

def normalize_nitro(mol):
    nitro_norm = rdChemReactions.ReactionFromSmarts("[N+:1](=[O:2])[O-:3]>>[N:1](:[O:2])(:[O+0:3])")
    new_mol = Chem.Mol(mol)
    while nitro_norm.RunReactantInPlace(new_mol):
        pass
    Chem.SanitizeMol(new_mol)
    return new_mol

def normalize_carboxylic_acid(mol):
    carboxylic_norm = rdChemReactions.ReactionFromSmarts("[C:1](=[O:2])[O:3]>>[C:1](:[O:2]):[O:3]")
    new_mol = Chem.Mol(mol)
    while carboxylic_norm.RunReactantInPlace(new_mol):
        pass
    Chem.SanitizeMol(new_mol)
    return new_mol


def calc_sigma_3d(mol, cutoff=0.1, gen3d=True, normalize=True):
    """
    Calculate the external rotational symmetry number for an RDKit molecule
    :param mol: input molecule
    :return: list of centers
    """
    # generate a 3d structure if necessary
    if gen3d:
        m1 = gen_3d(mol)
    else:
        m1 = Chem.Mol(mol)

    if normalize:
        m1 = normalize_nitro(m1)
        m1 = normalize_carboxylic_acid(m1)
    
    if m1 is None:
        # failed to generate a 3D structure
        print(f"Failed to generate a 3D structure for {Chem.MolToSmiles(mol)}",file=sys.stderr)
        sigma = -1
    else:
        # make a copy of the molecule
        m2 = Chem.Mol(m1)
        # generate a SMARTS to map the molecule onto itself
        query = Chem.MolFromSmarts(Chem.MolToSmiles(mol))
        match_list = m1.GetSubstructMatches(query, uniquify=False, maxMatches=10000)
        # loop over the symmetry matches and see if the RMS of the 3D overlap is less than 0.1A
        sigma = 0
        for i in range(0, len(match_list)):
            ap_list = list(zip(match_list[0], match_list[i]))
            rms = rdMolAlign.AlignMol(m1, m2, atomMap=ap_list)
            if rms < cutoff:
                sigma += 1
    return sigma


class SigmaCalculator:
    def __init__(self):
        buff = """attached_atoms	attached_types	max_nbrs	hyb	sigma
        2	1	0	SP	2
        2	1	0	SP2	2
        2	1	0	SP3	2
        2	2	0	SP	1
        2	2	0	SP2	1
        2	2	0	SP3	1
        3	1	0	SP2	6
        3	2	0	SP2	2
        3	3	0	SP2	1
        3	1	0	SP3	3
        3	2	0	SP3	1
        3	3	0	SP3	1
        4	1	0	SP3	12
        4	2	3	SP3	3
        4	2	2	SP3	2
        4	3	0	SP3	1
        4	4	0	SP3	1"""
        fs = StringIO(buff)
        self.table_df = pd.read_csv(fs, sep="\t")

        self.spiro_lookup = [-1,4,2,1,1]

        self.ether_pattern = Chem.MolFromSmarts("[#6]-O-[#6]")

    def has_ether(self,mol):
        return mol.HasSubstructMatch(self.ether_pattern)

    def calc_sigma_original(self, mol, normalize):
        mol = uru.get_largest_fragment(mol)
        if max_ring_size(mol) > 6 or num_contiguous_rotatable_bonds(mol) > 2:
            return 1
        if self.has_ether(mol):
            return 1

        if normalize:
            mol = normalize_nitro(mol)
            mol = normalize_carboxylic_acid(mol)
        
        # get a list of atoms in spiro fusions
        spiro_atoms = get_spiro_atoms(mol)
        # calculate atom invariants
        mol_invariants = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        # get the indices of the centers
        centers = get_centers(mol)
        # get the center atoms
        center_atms = [mol.GetAtomWithIdx(x) for x in centers]
        # loop over atoms and add up the contributions to sigma
        invariant_dict = defaultdict(list)
        for atm in center_atms:
            current_idx = atm.GetIdx()
            atm_invariant = mol_invariants[current_idx]
            # get the number of attached atoms
            num_attached_atoms = atm.GetDegree()
            # get the hybridization as a string
            hyb = str(atm.GetHybridization())
            # convert the neighbor potentials to a Pandas series
            nbr_invariants = [mol_invariants[x.GetIdx()] for x in atm.GetNeighbors()]
            nbr_series = pd.Series(nbr_invariants)
            # find the number of unique attached types
            num_attached_types = len(nbr_series.unique())
            if current_idx in spiro_atoms:
                if num_attached_types in range(1,5):
                    sigma = self.spiro_lookup[num_attached_types]
                else:
                    print(f"spiro error on {Chem.MolToSmiles(mol)}")
            else:
                # set max_attached_types = 0 when we don't need it
                max_attached_types = 0                
                # find the largest number of attached types, needed if
                # num_attached_atoms = 2 and num_attached_types = 2
                if hyb == 'SP3' and num_attached_atoms == 4 and num_attached_types == 2:
                    max_attached_types = max(nbr_series.value_counts().values)
                # look up sigma from the table that mirrors table 1 in the paper
                query = "hyb == @hyb and attached_atoms == @num_attached_atoms and attached_types == @num_attached_types"
                query += " and max_nbrs == @max_attached_types"
                res = self.table_df.query(query)
                if len(res) == 1:
                    sigma = res.sigma.values[0]
                else:
                    # set sigma to -100 in case of error
                    print(f"Error calculating {Chem.MolToSmiles(mol)}")
                    sigma = -100
            invariant_dict[atm_invariant].append(sigma)
            # print("attached = ",num_attached_atoms,"types = ",num_attached_types,"max = ",max_attached_types,hyb)
        return min([sum(v) for _, v in invariant_dict.items()])

    def calc_sigma_new(self, mol):
        mol = uru.get_largest_fragment(mol)
        if max_ring_size(mol) > 6 or num_contiguous_rotatable_bonds(mol) > 2:
            return 1
        if self.has_ether(mol):
            return 1
        sigma_lst = []
        for _ in range(0,10):
            sigma_val = calc_sigma_3d(mol)
            sigma_lst.append(sigma_val)
            if sigma_val == -1:
                break
        return max(sigma_lst)


def main():
    sigma_calculator = SigmaCalculator()
    smi = "c1cc2ccccc2[nH]1"
    smi = "ClC(Cl)(Cl)C1=CC=C(C=C1)C(Cl)(Cl)Cl"
    mol = Chem.MolFromSmiles(smi)
    print(sigma_calculator.calc_sigma_original(mol))
    print(sigma_calculator.calc_sigma_new(mol))


if __name__ == "__main__":
    main()
