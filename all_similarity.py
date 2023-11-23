#!/usr/bin/env python
# coding: utf-8

import time
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdmolops


import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from stqdm import stqdm

def tanimoto_similarity(query_smiles, all_mols):
    results = []
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
    for mol in stqdm(all_mols):
        similarity = DataStructs.TanimotoSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2, nBits=2048))
        result_entry = {
            'Query SMILE' : query_smiles,
            'Data SMILE' : mol,
            'Similarity' : similarity
        }
        results.append(result_entry)
    df1 = pd.DataFrame(results)
    return df1

def rdkit_similarity(query_smiles, all_mols):
    results = []
    query_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(query_smiles))
    for mol in stqdm(all_mols):
        rdkit_similarity = DataStructs.FingerprintSimilarity(query_fp, Chem.RDKFingerprint(Chem.MolFromSmiles(mol)))
        result_entry = {
            'Query SMILE' : query_smiles,
            'Data SMILE' : mol,
            'Similarity' : rdkit_similarity
        }
        results.append(result_entry)
    df1 = pd.DataFrame(results)
    return df1


def tversky_similarity(query_smiles, all_mols):
    results = []
    alpha = 0.5  # Weight for the query molecule
    beta = 0.5   # Weight for the reference molecule
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
    for mol in stqdm(all_mols):
        Chem.MolFromSmiles(mol)
        tversky_similarity = DataStructs.TverskySimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2, nBits=1024), alpha, beta)
        result_entry = {
            'Query SMILE' : query_smiles,
            'Data SMILE' : mol,
            'Similarity' : tversky_similarity
        }
        results.append(result_entry)
    df1 = pd.DataFrame(results)
    return df1

def geometric_similarity(query_smiles, all_mols):
    query_mol = Chem.MolFromSmiles(query_smiles)
    up_query_mol = rdmolops.AddHs(query_mol) 
    AllChem.EmbedMultipleConfs(up_query_mol, numConfs=10)
    for mol in all_mols:
        rmsd = []
        mol1 = rdmolops.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol1, numConfs=10)
        for confId1 in range(up_query_mol.GetNumConformers()):
            print("confId1",confId1)
            for confId2 in range(mol1.GetNumConformers()):
                print("confId2",confId2)
                rms = rdMolAlign.GetBestRMS(up_query_mol, mol1, prbCid=confId1, refCid=confId2)
                rmsd.append(rms)
        print(sum(rmsd))
        print(len(rmsd))
        geometric_similarity = len(rmsd) and sum(rmsd) / len(rmsd) or 0
        print(geometric_similarity)

def dice_similarity(query_smiles, all_mols):
    results = []
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
    for mol in stqdm(all_mols):
        dice_similarity = DataStructs.DiceSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2, nBits=1024))
        result_entry = {
            'Query SMILE' : query_smiles,
            'Data SMILE' : mol,
            'Similarity' : dice_similarity
        }
        results.append(result_entry)
    df1 = pd.DataFrame(results)
    return df1

def euclidian_similarity(query_smiles, all_mols):
    results = []
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
    for mol in stqdm(all_mols):
        euclidean_similarity = 1 - DataStructs.DiceSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2))
        result_entry = {
            'Query SMILE' : query_smiles,
            'Data SMILE' : mol,
            'Similarity' : euclidean_similarity
        }
        results.append(result_entry)
    df1 = pd.DataFrame(results)
    return df1

def cosine_similarity(query_smiles, all_mols):
    results = []
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
    for mol in stqdm(all_mols):
        cosine_similarity = DataStructs.CosineSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2, nBits=1024))
        result_entry = {
            'Query SMILE' : query_smiles,
            'Data SMILE' : mol,
            'Similarity' : cosine_similarity
        }
        results.append(result_entry)
    df1 = pd.DataFrame(results)
    return df1

def rogot_goldberg_similarity(query_smiles, all_mols):
    results = []
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
    for mol in stqdm(all_mols):
        rogot_goldberg_similarity = DataStructs.FingerprintSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2, nBits=1024), metric=DataStructs.DiceSimilarity)
        result_entry = {
            'Query SMILE' : query_smiles,
            'Data SMILE' : mol,
            'Similarity' : rogot_goldberg_similarity
        }
        results.append(result_entry)
    df1 = pd.DataFrame(results)
    return df1

def MACCS_keys_similarity(query_smiles, all_mols):
    results = []
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMACCSKeysFingerprint(query_mol)
    for mol in stqdm(all_mols):
        MACCS_keys_similarity = DataStructs.FingerprintSimilarity(query_fp, AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(mol)))
        result_entry = {
            'Query SMILE' : query_smiles,
            'Data SMILE' : mol,
            'Similarity' : MACCS_keys_similarity
        }
        results.append(result_entry)
    df1 = pd.DataFrame(results)
    return df1   

def manhattan_similarity(query_smiles, all_mols):
    results = []
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
    for mol in stqdm(all_mols):
        manhattan_similarity = 1 - DataStructs.FingerprintSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2, nBits=1024), metric=DataStructs.DiceSimilarity)
        result_entry = {
            'Query SMILE' : query_smiles,
            'Data SMILE' : mol,
            'Similarity' : manhattan_similarity
        }
        results.append(result_entry)
    df1 = pd.DataFrame(results)
    return df1


start_time = time.time()
#exp_sdf x[x.columns[0]]

st.title('Molecular similarity')

input_smile = st.text_input('Molecular SMILES', 'c1ccccc1')
st.write('The current SMILES is', input_smile)

st.write("Tanimoto Similarity results: ")
query_smiles=input_smile
all_smiles = pd.read_csv('data/pubchem_100000_only_cleaned_smiles.csv', header=None)
# all_smiles = pd.read_csv('data/temp.csv', header=None)

all_mols =  all_smiles[all_smiles.columns[0]]
with st.spinner('Checking Tanimoto Similarity ...'):
    x = tanimoto_similarity(query_smiles, all_mols)
    y = x.nlargest(10, 'Similarity')
st.dataframe(y)

end_time = time.time()
elasped_time = end_time - start_time
st.write("Total time taken tanimoto_similarity:", elasped_time)
##########################

start_time1 = time.time()
st.write("RDKit Similarity results: ")
with st.spinner('Checking RDKit Similarity ...'):
    x = rdkit_similarity(query_smiles, all_mols)
    y = x.nlargest(10, 'Similarity')
st.dataframe(y)

end_time = time.time()
elasped_time = end_time - start_time1
st.write("Total time taken rdkit_similarity:", elasped_time)
# ##########################

start_time2 = time.time()
st.write("Tversky Similarity results: ")
with st.spinner('Checking Tversky Similarity ...'):
    x = tversky_similarity(query_smiles, all_mols)
    y = x.nlargest(10, 'Similarity')
st.dataframe(y)

end_time = time.time()
elasped_time = end_time - start_time2
st.write("Total time taken tversky_similarity:", elasped_time)
# # ##########################

start_time3 = time.time()
st.write("Euclidian Similarity results: ")
with st.spinner('Checking Euclidian Similarity ...'):
    x = euclidian_similarity(query_smiles, all_mols)
    y = x.nlargest(10, 'Similarity')
st.dataframe(y)

end_time = time.time()
elasped_time = end_time - start_time3
st.write("Total time taken euclidian_similarity:", elasped_time)
# # ##########################

start_time4 = time.time()
st.write("Dice Similarity results: ")
with st.spinner('Checking Dice Similarity ...'):
    x = dice_similarity(query_smiles, all_mols)
    y = x.nlargest(10, 'Similarity')
st.dataframe(y)

end_time = time.time()
elasped_time = end_time - start_time4
st.write("Total time taken dice_similarity:", elasped_time)
# ##########################

start_time5 = time.time()
st.write("Cosine Similarity results: ")
with st.spinner('Checking Cosine Similarity ...'):
    x = cosine_similarity(query_smiles, all_mols)
    y = x.nlargest(10, 'Similarity')
st.dataframe(y)

end_time = time.time()
elasped_time = end_time - start_time5
st.write("Total time taken cosine_similarity:", elasped_time)
# ##########################

start_time6 = time.time()
st.write("Rogot Goldberg Similarity results: ")
with st.spinner('Checking Cosine Similarity ...'):
    x = rogot_goldberg_similarity(query_smiles, all_mols)
    y = x.nlargest(10, 'Similarity')
st.dataframe(y)

end_time = time.time()
elasped_time = end_time - start_time6
st.write("Total time taken rogot_goldberg_similarity:", elasped_time)
# ##########################

start_time7 = time.time()
st.write("MACCS Keys Similarity results: ")
with st.spinner('Checking MACCS Keys Similarity ...'):
    x = MACCS_keys_similarity(query_smiles, all_mols)
    y = x.nlargest(10, 'Similarity')
st.dataframe(y)

end_time = time.time()
elasped_time = end_time - start_time7
st.write("Total time taken MACCS_keys_similarity:", elasped_time)
# ############################


start_time8 = time.time()
st.write("Manhattan Similarity results: ")
with st.spinner('Checking Manhattan Similarity ...'):
    x = manhattan_similarity(query_smiles, all_mols)
    y = x.nlargest(10, 'Similarity')
st.dataframe(y)

end_time = time.time()
elasped_time = end_time - start_time8
st.write("Total time taken manhattan_similarity:", elasped_time)

# ##########################

end_time = time.time()
elasped_time = end_time - start_time
print("Total time taken :", elasped_time)





