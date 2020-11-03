# UV-Vis peak prediction from topological features

MLP regression model for UV-Vis peak prediction from the topological 
features of conjugated sub graphs of molecules using RDKit and 
Scikit-learn. [Full report is available here](https://htleppan.github.io/UVVis-Topo-MLP/UVVis-Topo-MLP.html).

- preprocess_data.py - Extract topological fingerprints from SMILES strings (RDKit)
- train_mlp.py - Training an MLP regressor (Scikit-learn)
- paper_allDB.csv - Input data from https://www.nature.com/articles/s41597-019-0306-0
