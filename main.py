import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['RDKIT_CANVAS'] = '1'

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)
except ImportError:
    st.error("Please install RDKit: pip install rdkit-pypi")
    st.stop()

# Simple CSS without backgrounds
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    [data-testid="metric-container"] {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    </style>
""", unsafe_allow_html=True)

def create_molecular_graph(smiles):
    """Convert SMILES to molecular graph using NetworkX"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        rdDepictor.Compute2DCoords(mol)
        G = nx.Graph()
        
        # Add nodes (atoms)
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                      atomic_num=atom.GetAtomicNum(),
                      symbol=atom.GetSymbol(),
                      formal_charge=atom.GetFormalCharge(),
                      implicit_valence=atom.GetImplicitValence(),
                      is_aromatic=atom.GetIsAromatic())
        
        # Add edges (bonds)
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                      bond.GetEndAtomIdx(),
                      bond_type=bond.GetBondTypeAsDouble(),
                      is_aromatic=bond.GetIsAromatic())
        
        return G, mol
    except Exception as e:
        st.error(f"Error creating molecular graph: {str(e)}")
        return None, None

def visualize_graph(G, mol):
    """Create visualization of the molecular graph"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot molecular structure
        img = Draw.MolToImage(mol, size=(400, 400))
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Molecular Structure', pad=20)
        
        # Plot graph representation
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Color map for atoms
        atom_colors = {
            1: '#FFFFFF',   # H - White
            6: '#808080',   # C - Gray
            7: '#0000FF',   # N - Blue
            8: '#FF0000',   # O - Red
            9: '#90EE90',   # F - Light green
            15: '#FFA500',  # P - Orange
            16: '#FFFF00',  # S - Yellow
            17: '#00FF00',  # Cl - Green
            35: '#A52A2A',  # Br - Brown
            53: '#800080'   # I - Purple
        }
        
        # Draw nodes
        node_colors = [atom_colors.get(G.nodes[node]['atomic_num'], '#FFFFFF') 
                      for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=1000, alpha=0.7, ax=ax2)
        
        # Draw edges
        edge_weights = [G[u][v]['bond_type'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                             edge_color='gray', alpha=0.5, ax=ax2)
        
        # Add labels
        labels = {i: G.nodes[i]['symbol'] for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, 
                              font_weight='bold', ax=ax2)
        
        ax2.set_title('Graph Representation', pad=20)
        ax2.axis('off')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error visualizing graph: {str(e)}")
        return None

def get_molecule_properties(mol):
    """Calculate molecular properties"""
    try:
        properties = {
            'Molecular Weight': Descriptors.ExactMolWt(mol),
            'Number of Atoms': mol.GetNumAtoms(),
            'Number of Bonds': mol.GetNumBonds(),
            'Number of Rings': Chem.rdMolDescriptors.CalcNumRings(mol),
            'TPSA': Descriptors.TPSA(mol),
            'LogP': Descriptors.MolLogP(mol),
            'H-Bond Donors': Descriptors.NumHDonors(mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol)
        }
        return properties
    except Exception as e:
        st.error(f"Error calculating properties: {str(e)}")
        return {}

# App header
st.title("GraphMol Nexus")
st.write("Convert molecular structures into interactive graph visualizations")

# Input options
st.sidebar.header("Input Options")
input_type = st.sidebar.selectbox("Choose input method", 
                                 ["SMILES", "Example Molecules"])

if input_type == "SMILES":
    smiles_input = st.sidebar.text_input("Enter SMILES string:", 
                                        "CC(=O)OC1=CC=CC=C1C(=O)O")
else:
    example_molecules = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)[C@H](C)C(=O)O",
        "Benzene": "c1ccccc1",
        "Ethanol": "CCO",
        "Glucose": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O"
    }
    selected_molecule = st.sidebar.selectbox("Choose a molecule:", 
                                           list(example_molecules.keys()))
    smiles_input = example_molecules[selected_molecule]

st.sidebar.text_area("Current SMILES:", smiles_input, height=100)

# Process and display
if smiles_input:
    G, mol = create_molecular_graph(smiles_input)
    
    if G is None:
        st.error("Invalid SMILES string! Please check your input.")
    else:
        # Display visualization
        fig = visualize_graph(G, mol)
        if fig is not None:
            st.pyplot(fig)
        
        # Display properties
        st.markdown('<h2 style="color: #4CAF50;">Molecular Properties</h2>', 
                   unsafe_allow_html=True)
        properties = get_molecule_properties(mol)
        
        if properties:
            cols = st.columns(3)
            for idx, (prop, value) in enumerate(properties.items()):
                with cols[idx % 3]:
                    st.metric(
                        prop,
                        f"{value:.2f}" if isinstance(value, float) else value
                    )
            
            st.markdown('<h2 style="color: #4CAF50;">Graph Properties</h2>', 
                       unsafe_allow_html=True)
            graph_props = {
                "Number of Nodes": G.number_of_nodes(),
                "Number of Edges": G.number_of_edges(),
                "Average Degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
                "Graph Density": nx.density(G)
            }
            
            cols = st.columns(2)
            for idx, (prop, value) in enumerate(graph_props.items()):
                with cols[idx % 2]:
                    st.metric(
                        prop,
                        f"{value:.2f}" if isinstance(value, float) else value
                    )

# Sidebar information
st.sidebar.markdown("""
## How to Use
1. Choose input method (SMILES or Example Molecules)
2. If using SMILES, enter a valid SMILES string
3. View the molecular structure and graph representation
4. Explore molecular and graph properties

## About SMILES
SMILES (Simplified Molecular Input Line Entry System) is a specification for 
describing the structure of chemical molecules using short ASCII strings.
""")
