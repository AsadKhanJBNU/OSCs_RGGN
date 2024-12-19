# Import necessary libraries for PyTorch and PyTorch Geometric
import torch
import os
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from models.Molecular_predictor_resgatedgraphconv import MolPredictor
from models.regression_train_test import train, test, predictingSingle
from torch_geometric.utils import from_smiles
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
import time


# Set CUDA device order and visibility
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Seed for reproducibility
seed = 120

# Set device to GPU if available, else CPU
if torch.cuda.is_available():  
    device = "cuda:4"
    torch.cuda.manual_seed_all(seed)
else:  
    device = "cpu"
device = "cpu"

# Custom dataset class for handling molecular data
class Molecule_data(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', y=None, transform=None,
                 pre_transform=None, smiles=None):
       
        super(Molecule_data, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        # Load processed data if it exists, otherwise process raw data
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process(smiles)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Defines raw file names (not used here)."""
        pass

    @property
    def processed_file_names(self):
        """Specifies the name of the processed file."""
        return [self.dataset + '.pt']

    def download(self):
        """Placeholder for downloading data (not used)."""
        pass

    def _download(self):
        """Internal placeholder for download logic."""
        pass

    def _process(self):
        """Ensures the processed directory exists."""
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, smiles):
        """
        Processes SMILES strings into molecular graph data.

        Parameters:
        -----------
        smiles : list
            List of SMILES strings to convert into graph data.
        """
        data_list = []
        for i in range(len(smiles)):
            smile = smiles[i]
            data = from_smiles(smile)
            data.x = (data.x).type(torch.FloatTensor)
            data.edge_attr = (data.edge_attr).type(torch.FloatTensor)
            data.smile_fingerprint = None
            graph = data
            
            data_list.append(graph)

        # Apply filters and transforms if specified
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Function to predict molecular properties
def Predictions(smilesList, targetToPredicate):
    transform = T.Compose([T.NormalizeFeatures(['x', 'edge_attr'])])
    
    test_data_set = 'test_data_set' + str(time.time_ns())
    test_data = Molecule_data(root='data', dataset=test_data_set, y=None,
                              smiles=smilesList, transform=transform)
    noveltest_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    
    model = MolPredictor().to('cpu')  # Load model onto CPU
    savepath = 'molecule_property_' + targetToPredicate + "/"
    model_file_name = 'saved_models/' + savepath + 'model.model'
    checkpoint = torch.load(model_file_name, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    prediction = predictingSingle(noveltest_loader, model, 'cpu')
    return prediction

# Create a radar (spider) chart
def create_spider_chart(predictions, title="Predictions"):
    # Prepare data for the spider chart
    categories = list(predictions.keys())
    values = list(predictions.values())
    
    # Ensure the chart closes by repeating first values
    values += values[:1]
    categories += categories[:1]
    
    # Calculate angles for each category
    angles = np.linspace(0, 2 * np.pi, len(categories) - 1, endpoint=False).tolist()
    angles += angles[:1]  # Repeat first angle to close the polygon
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Fill the polygon
    ax.fill(angles, values, color='blue', alpha=0.25)
    
    # Draw the outline
    ax.plot(angles, values, color='blue', linewidth=2)
    
    # Customize the chart
    ax.set_yticks([])  # Remove radial gridlines
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=10, color="black")
    ax.set_title(title, fontsize=16, pad=20)
    
    return fig

# Convert SMILES strings to molecular structure images
def smilesToMolecularStructure(smiles):
    images = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            img = Draw.MolToImage(mol)
            images.append(img)
        else:
            images.append(None)

    return images

# Main function for making predictions
def makePredictions(inputSmiles):

    targets_to_predicates = ['HOMO', 'LUMO', 'E(S1)', 'f(S1)', 'E(S2)', 'f(S2)', 
                             'E(S3)', 'f(S3)', 'E(T1)', 'E(T2)', 'E(T3)']
    predictions_array = {}
    for target in targets_to_predicates:
        predictions = Predictions(inputSmiles, target)
        predictions_array[target] = predictions
    return predictions_array

# Example usage
smileslist = ['c1cc2cc(ccc2nc1)O']
predictions_result = makePredictions(smileslist)
print(predictions_result)

# Create the spider chart for the predictions
fig = create_spider_chart(predictions_result)

# Generate molecular structure images
smilesImages = smilesToMolecularStructure(smileslist)
print(smilesImages)
for img in smilesImages:
    if img:
        img.show()
