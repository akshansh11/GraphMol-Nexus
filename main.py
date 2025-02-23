import rdkit
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# Import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
except ImportError:
    st.error("Please install RDKit using: pip install rdkit")
    st.stop()

# Set page configuration
st.set_page_config(page_title="MoleculeVortex", layout="wide")

# Custom CSS for futuristic styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e1e1e1;
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
    .css-1d391kg {  /* Sidebar */
        background: rgba(22, 33, 62, 0.9);
    }
    .stTitle {
        text-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
    }
    h2 {
        color: #4CAF50;
        text-shadow: 0 0 5px rgba(76, 175, 80, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def create_molecular_graph(smiles):
    """Convert SMILES to molecular graph using NetworkX"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (atoms)
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                      atomic_num=atom.GetAtomicNum(),
                      symbol=atom.GetSymbol(),
                      formal_charge=atom.GetFormalCharge(),
                      implicit_valence=atom.GetImplicitValence())
        
        # Add edges (bonds)
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                      bond.GetEndAtomIdx(),
                      bond_type=bond.GetBondTypeAsDouble())
        
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
        img = Draw.MolToImage(mol)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Molecular Structure', pad=20, fontsize=14, color='white')
        ax1.set_facecolor('#1a1a2e')
        
        # Plot graph representation
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes with different colors based on atomic number
        node_colors = [plt.cm.plasma(atom['atomic_num'] / 20) for _, atom in G.nodes(data=True)]
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
            'Molecular Weight': Chem.Descriptors.ExactMolWt(mol),
            'Number of Atoms': mol.GetNumAtoms(),
            'Number of Bonds': mol.GetNumBonds(),
            'Number of Rings': Chem.rdMolDescriptors.CalcNumRings(mol),
            'TPSA': Chem.Descriptors.TPSA(mol),
            'LogP': Chem.Descriptors.MolLogP(mol)
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
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)[C@H](C)C(=O)O"
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
                    st.metric(prop, f"{value:.2f}" if isinstance(value, float) else value)
            
            # Display graph properties
            st.header("Graph Properties")
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
                    st.metric(prop, f"{value:.2f}" if isinstance(value, float) else value)

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
