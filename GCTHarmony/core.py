from owlready2 import get_ontology
import re
import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt
import rdflib
import os
from matplotlib.lines import Line2D
from scipy.spatial.distance import cosine
from openai import OpenAI
import argparse

client: OpenAI = None

data_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data')
)

# Read all cell type ontologies through OWL link
ontology_url = os.path.join(data_dir, 'cl_20250129.owl')
cl_ontology = get_ontology(ontology_url).load()
labels_all = []
pattern = re.compile(r'CL')
for cls in cl_ontology.classes():
    label = cls.label
    if label:
        if re.match(r'CL',cls.name):
            labels_all.append(label[0])

# Read human cell altlas dataset
with open(os.path.join(data_dir,'HRA_cell_type.txt'), 'r') as file:
    hra_cell_type_file = [line.strip() for line in file]
hra_cell_type = []
for i in labels_all:
    if i.casefold() in [x.casefold() for x in hra_cell_type_file]:
        hra_cell_type.append(str(i))
hra_cell_type = sorted(hra_cell_type)

# Read 2 embeddings files
embeddings = np.load(os.path.join(data_dir,'embeddings_all_CL.npy'))
embeddings_response = np.load(os.path.join(data_dir,'embeddings_all_CL_response.npy'))

embeddings_all_dict = {}
for cl, eb in zip(labels_all, embeddings):
    embeddings_all_dict[cl] = eb

embeddings_response_all_dict = {}
for cl, eb in zip(labels_all, embeddings_response):
    embeddings_response_all_dict[cl] = eb

embeddings_hra = []
embeddings_response_hra = []
for h in hra_cell_type:
    for c in labels_all:
        if h.casefold() == c.casefold():
            embeddings_hra.append(embeddings_all_dict[c])
            embeddings_response_hra.append(embeddings_response_all_dict[c])
embeddings_hra = np.array(embeddings_hra)
embeddings_response_hra = np.array(embeddings_response_hra)

### important functions ###
def _make_client(api_key):
    if not api_key:
        raise ValueError("You must pass your OpenAI API key via `api_key`.")
    return OpenAI(api_key=api_key)

def _get_embedding(text, client, model="text-embedding-3-large"):
    txt = text.replace("\n", " ")
    return client.embeddings.create(input=[txt], model=model).data[0].embedding

def find_closest_cell_type(name,client,mode = 'simple', background = 'CL'):
    target = _get_embedding(name,client,"text-embedding-3-large")
    closest_name = None
    closest_distance = float('inf')
    
    if mode == 'simple' and background == 'CL':
        eb = embeddings 
        labels = labels_all

    elif mode == 'simple' and background == 'HRA':
        eb = embeddings_hra 
        labels = hra_cell_type
        
    elif mode == 'complex' and background == 'CL':
        eb = embeddings_response 
        labels = labels_all

    elif mode == 'complex' and background == 'HRA':
        eb = embeddings_response_hra 
        labels = hra_cell_type
    
    for cell_type, embedding in zip(labels, eb):
        distance = cosine(target, embedding)
        if distance < closest_distance:
            closest_distance = distance
            closest_name = cell_type
            
    return closest_name, closest_distance

def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",  # Specify the GPT model you want to use
        messages=[
            {"role": "user", "content": f'Please use 1 sentences to describe cell type: {prompt}'}
        ]
    )
    return response.choices[0].message.content.strip()

def GCTHarmony(input_cts,mode = 'simple',background = 'HRA',api_key=None):

    client = _make_client(api_key)
    out = []
    for ct in input_cts:
        if mode == 'simple':
            mapped, _ = find_closest_cell_type(ct, client, mode, background)

        elif mode == 'complex':
            descr     = get_response(ct, client)
            mapped, _ = find_closest_cell_type(descr, client, mode, background)
        out.append(mapped)
    return out

if __name__ == '__main__':
    pass
