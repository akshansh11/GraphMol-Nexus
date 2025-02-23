import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['RDKIT_CANVAS'] = '1'  # Enable RDKit canvas

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)
except ImportError:
    st.error("""RDKit import failed. Please run these commands:
    ```
    pip install rdkit-pypi
    ```
    """)
    st.stop()

# Set page configuration
st.set_page_config(page_title="MoleculeVortex", layout="wide")

# Custom CSS for better visibility
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.3);
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(76, 175, 80, 0.5);
    }
    .css-1d391kg {
        background: rgba(22, 33, 62, 0.9);
    }
    .stTitle {
        text-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
        color: white !important;
    }
    h1, h2, h3 {
        color: #4CAF50 !important;
        text-shadow: 0 0 5px rgba(76, 175, 80, 0.2);
    }
    /* Make ALL metric text white and bright */
    div[data-testid="stMetricValue"] > div {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: white !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        opacity: 0.9;
    }
    /* Make metric cards more visible */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    div[data-testid="metric-container"]:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(76, 175, 80, 0.2);
        margin: 5px;
    }
    .stMarkdown, .stText {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_molecular_graph(smiles):
    """Convert SMILES to molecular graph using NetworkX"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        # Generate 2D coordinates for the molecule
        rdDepictor.Compute2DCoords(mol)
        
        # Create graph
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
    """Create an interactive visualization of the molecular graph"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.patch.set_facecolor('#1a1a2e')
        
        # Plot molecular structure
        drawer = Draw.rdDepictor.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(400, 400))
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Molecular Structure', pad=20, fontsize=14, color='white')
        ax1.set_facecolor('#1a1a2e')
        
        # Plot graph representation
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create color map for atoms
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
        
        # Draw nodes with different colors based on atomic number
        node_colors = [atom_colors.get(G.nodes[node]['atomic_num'], '#FFFFFF') 
                      for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=1000, alpha=0.7, ax=ax2)
        
        # Draw edges with varying thickness based on bond type
        edge_weights = [G[u][v]['bond_type'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, 
                             edge_color='lightblue', alpha=0.5, ax=ax2)
        
        # Add labels
        labels = {i: G.nodes[i]['symbol'] for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, 
                              font_weight='bold', ax=ax2)
        
        ax2.set_title('Graph Representation', pad=20, fontsize=14, color='white')
        ax2.set_facecolor('#1a1a2e')
        ax2.axis('off')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error visualizing graph: {str(e)}")
        return None

def get_molecule_properties(mol):
    """Calculate and return basic molecular properties"""
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

# Main Streamlit app
st.title("🌀 MoleculeVortex")
st.write("Convert molecular structures into interactive graph visualizations")

# Input section
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

# Display current SMILES
st.sidebar.text_area("Current SMILES:", smiles_input, height=100)

# Convert and visualize
if smiles_input:
    G, mol = create_molecular_graph(smiles_input)
    
    if G is None:
        st.error("Invalid SMILES string! Please check your input.")
    else:
        # Display visualization
        fig = visualize_graph(G, mol)
        if fig is not None:
            st.pyplot(fig)
        
        # Display molecular properties
        st.header("Molecular Properties")
        properties = get_molecule_properties(mol)
        
        if properties:
            # Create three columns for properties
            cols = st.columns(3)
            for idx, (prop, value) in enumerate(properties.items()):
                with cols[idx % 3]:
                    st.metric(
                        prop,
                        f"{value:.2f}" if isinstance(value, float) else value,
                        delta=None,
                        help=f"Value for {prop}"
                    )
            
            # Display graph properties
            st.markdown('<h2 style="color: #4CAF50;">Graph Properties</h2>', unsafe_allow_html=True)
            graph_props = {
                "Number of Nodes": G.number_of_nodes(),
                "Number of Edges": G.number_of_edges(),
                "Average Degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
                "Graph Density": nx.density(G)
            }
            
            # Create two columns for graph properties
            cols = st.columns(2)
            for idx, (prop, value) in enumerate(graph_props.items()):
                with cols[idx % 2]:
                    st.metric(
                        prop,
                        f"{value:.2f}" if isinstance(value, float) else value,
                        delta=None,
                        help=f"Value for {prop}"
                    )

# Add information about usage
st.sidebar.markdown("""
## How to Use
1. Choose input method (SMILES or Example Molecules)
2. If using SMILES, enter a valid SMILES string
3. View the molecular structure and graph representation
4. Explore molecular and graph properties

## About SMILES
SMILES (Simplified Molecular Input Line Entry System) is a specification for describing the structure of chemical molecules using short ASCII strings.
""")
