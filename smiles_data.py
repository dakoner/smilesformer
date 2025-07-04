import gzip
import random

# smiles_data = [
#     "CC(=O)OC1=CC=CC=C1C(=O)O", # Aspirin
#     "C1=CC=C(C=C1)C(C(C(=O)O)N)O", # A random amino acid derivative
#     "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", # Ibuprofen
#     "C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C=C(C=C3)N", # An aminoanthraquinone
#     "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C4=CN=CC=C4", # Imatinib
#     "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
#     "C(C(=O)O)N", # Glycine
#     "CC(C(=O)O)N", # Alanine
#     "C1=CN=CN1", # Imidazole
#     "C1CCOC1", # Tetrahydrofuran
#     "C1=CC=CS1", # Thiophene
#     "CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC", # Melatonin
# ]
# smiles_data *= 10 # Repeat data to simulate a larger dataset
# random.shuffle(smiles_data)

# Add more data for better training
gzip_file_path = "dataJ_250k_rndm_zinc_drugs_clean.txt.gz"
f = gzip.open(gzip_file_path, 'rt')
lines = f.readlines()
smiles_data = [ line.strip() for line in lines ]