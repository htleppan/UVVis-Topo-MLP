import numpy as np
import pandas as pd 
from rdkit import Chem

def get_fingerprint_from_smiles(smi, path_length = 4, path_is_ordered = True, both_path_directions = True, verbose = False):

    mol = Chem.MolFromSmiles(smi)
    if mol == None:
        return None
    if len(mol.GetBonds()) < 4:
        return None

    Chem.rdmolops.RemoveHs(mol)
    
    # DFS Search to find all conjugated bonds
    # Results in a list of sets of atom index pairs in conjugated_bonds
    # e.g. conjugated_bonds = [{0, 1}, {1, 2}, {2, 3}]
    conjugated_bonds = []
    visited_bonds = []
    bond_stack = [mol.GetBonds()[0]]

    while(len(bond_stack) > 0):
        this_bond = bond_stack.pop()
        begin_atom_idx = this_bond.GetBeginAtom().GetIdx()
        end_atom_idx = this_bond.GetEndAtom().GetIdx()
        
        visited_bonds.append({begin_atom_idx, end_atom_idx})

        if this_bond.GetIsConjugated():
            conjugated_bonds.append({begin_atom_idx, end_atom_idx})

        for bond in mol.GetAtomWithIdx(end_atom_idx).GetBonds():
            if {bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()} not in visited_bonds:
                bond_stack.append(bond)

    # Return None if no conjugated paths were found
    if len(conjugated_bonds) == 0:
        return None

    if verbose:
        print("Conjugated bonds:")
        print(conjugated_bonds)
                
    
    ### Generate connected conjugation sets from a list of atom pairs
    ### e.g. [{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {0, 5}, {6, 7}, {7, 8}]
    ###  --> [{0,1,2,3,4,5}, {6,7,8}]
    
    conjugated_sets = []
    for bond in conjugated_bonds:
        appended_to_set = False
        
        # Check if this bond can be added to existing conjugated set
        for conj_set in conjugated_sets:
            if len(bond.intersection(conj_set)) > 0:
                conj_set.update(bond)
                appended_to_set = True
        
        # Else create new conjugated set
        if appended_to_set == False:
            conjugated_sets.append(bond)
            
    if verbose:
        print("Conjugated sets:")
        print(conjugated_sets)

        
        
    ### Find the largest conjugation set; this will be used for the lambda prediction
    ### Return None if too short (< 4 atoms)

    set_sizes = [len(x) for x in conjugated_sets]
    max_conjugated_set = conjugated_sets[np.argmax(set_sizes)]
    
    if len(max_conjugated_set) < 4:
        if verbose:
            print("Too small max. conjugated set")
        return None
    if verbose:
        print("Max set:")
        print(max_conjugated_set)

    
    ### Find all substituent atoms connected directly to the conjugated sub graph
    
    substituent_atoms = set()
    for atom_idx in max_conjugated_set:
        for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
            substituent_atoms.add(bond.GetBeginAtom().GetIdx())
            substituent_atoms.add(bond.GetEndAtom().GetIdx())
    max_conjugated_set.update(substituent_atoms)


    ### Generate all atom paths of size path_length within the molecule
    ### retain only if all atoms in the route belong to max_conjugated_set
    
    atom_paths = []
    for atom_idx in max_conjugated_set:
        paths = Chem.rdmolops.FindAllPathsOfLengthN(mol, path_length, rootedAtAtom=atom_idx, useBonds = False)
        for p in paths:
            path_list = []
            for path_atom in p:
                path_list.append(path_atom)
                
            # Check if all atoms exist in max_conjugated_set
            if len(max_conjugated_set.intersection(set(path_list))) == len(path_list):
                atom_paths.append(path_list)

    if verbose:
        print("Atom paths: ")
        print(atom_paths)

        
        
    ### Generate atom path strings
    atom_paths_dict = {}
    duplicate_check_list = []
    for path in atom_paths:
        atom_string_list = []
        duplicate_check = []
        for atom_idx in path:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_string_list.append(str(atom.GetSymbol()) + " " + str(atom.GetHybridization()) + " " + str(atom.GetIsAromatic()) + " " + str(atom.GetExplicitValence()))
            duplicate_check.append(atom_idx)
        if both_path_directions == False:
            if duplicate_check in duplicate_check_list or list(reversed(duplicate_check)) in duplicate_check_list:
                continue
            duplicate_check_list.append(duplicate_check)
            
        if path_is_ordered == False:
            atom_string_list.sort()
        if both_path_directions == False:
            needs_reverse = False
            for i in range(0, len(atom_string_list)):
                if atom_string_list[i] > atom_string_list[-(i + 1)]:
                    needs_reverse = True
                else:
                    break
            if needs_reverse:
                if verbose:
                    print("Before combining both directions " + str(atom_string_list))

                tmp_string = []
                for i in range(0, len(atom_string_list)):
                    tmp_string.append(atom_string_list[-(i + 1)])
                atom_string_list = tmp_string
                if verbose:
                    print("After combining both directions " + str(atom_string_list))


        path_string = " . ".join(atom_string_list)
        if path_string not in atom_paths_dict:
            atom_paths_dict[path_string] = 1
        else:
            atom_paths_dict[path_string] += 1

    if verbose:
        print("Conjugated atom paths: ")
        for ap in atom_paths_dict:
            print(str(ap) + ": " + str(atom_paths_dict[ap]))
                
    return atom_paths_dict



### Read UV-Vis data; get SMILES and sTDA columns only
df = pd.read_csv("paper_allDB.csv", header = 0, usecols = [0, 1])
df.columns = ["SMI", "sTDA (nm)"]
### Retain rows with Lambda in range of [200, 800] 
df = df[df['sTDA (nm)'].notnull()]
df = df[df["sTDA (nm)"] <= 800]
df = df[df["sTDA (nm)"] >= 200]


### Get fingerprint and add to data frame; discard rows where fingerprint generation failed
df["fingerprint"] = [get_fingerprint_from_smiles(x, path_length = 4, both_path_directions=False, path_is_ordered = True) for x in df["SMI"]]
df = df[df["fingerprint"].notnull()]

### Generate global path key list
allKeys = set()
k = df["fingerprint"].apply(lambda x: allKeys.update(x.keys()))
allKeys = list(allKeys)

### Generate input vectors based on df["fingerprint"] and allKeys
### Scale input data to [0, 1]
input_data = np.zeros((df.shape[0], len(allKeys)))
fps = df.loc[:, "fingerprint"]
for m in range(0, fps.shape[0]):
    for k in fps.iloc[m]:
        input_data[m, allKeys.index(k)] = fps.iloc[m][k]
print("Input data shape (entries, vecdim): " + str(input_data.shape))
input_data = (input_data - np.min(input_data, axis = 0)) / (np.max(input_data, axis = 0) - np.min(input_data, axis = 0))

### Generate output data
output_data = df[["sTDA (nm)"]].to_numpy()
print("Output data shape: " + str(output_data.shape))

### Get SMILES entries
smiles = df[["SMI"]]

### Save all
np.savetxt("input_data.csv", input_data, delimiter=",")
np.savetxt("output_data.csv", output_data, delimiter=",")
smiles.to_csv("smiles.csv")
