from owlready2 import get_ontology, ThingClass
import re
import os
from collections import deque

data_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data')
)
ONTOLOGY_PATH = os.path.join(data_dir, 'cl_20250129.owl')


################ merge to parent cell type by constant hierarchy levels #########################
def traverse_subclasses(cls, all_cell_types, level):
   
    label = cls.label[0] if cls.label else cls.name
    all_cell_types[str(label)] = level
    
    for subclass in cls.subclasses():
        traverse_subclasses(subclass, all_cell_types, level + 1)


def CLhierarchy():
    
    onto = get_ontology(ONTOLOGY_PATH).load()
    cell_class = onto.search_one(label="cell")
    all_cell_types = {}
    
    if cell_class is not None:
        traverse_subclasses(cell_class, all_cell_types, level=0)
        
    return all_cell_types
    
def get_parent_cell_types_by_name(cell_names, all_cell_types, hierarchy_level=None):
    onto = get_ontology(ONTOLOGY_PATH).load()
    parent_mapping = {}
    
    if hierarchy_level is None:
        hierarchy_level = min([all_cell_types[c] for c in cell_names])

    for cell_name in cell_names:
        try:
            level = all_cell_types[cell_name]
            cell_class = next((cls for cls in onto.classes() if hasattr(cls, "label") and cell_name in cls.label), None)
            
            if level <= hierarchy_level:
                parent_mapping[cell_name] = [cell_name]
                
            while level > hierarchy_level:
                
                if cell_class:
                    parents = cell_class.is_a  
                    parent_labels = [p.label[0] for p in parents if hasattr(p, "label") and p.label]
                    parent_mapping[cell_name] = parent_labels if parent_labels else ["No parent found"]
                    cell_class = parents[0]
                    level = all_cell_types[str(cell_class.label[0])]
                else:
                    parent_mapping[cell_name] = ["Not found in ontology"]
                
        except Exception as e:
            parent_mapping[cell_name] = [f"Error: {str(e)}"]

    return parent_mapping

################ merge to parent cell type exising in the dataset #########################
def get_all_subclasses(cls):
    subclasses = list(cls.subclasses()) 
    all_subclasses = subclasses[:]  
    all_subclasses.append(cls) 
    for subclass in subclasses:
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses

def get_all_subclasses(cls):
    subclasses = list(cls.subclasses())  
    all_subclasses = subclasses[:]  
    all_subclasses.append(cls) 
    for subclass in subclasses:
        all_subclasses.extend(get_all_subclasses(subclass))  
    return all_subclasses

def get_labels(clas):
    labels_all = []
    pattern = re.compile(r'CL')
    onto = get_ontology(ONTOLOGY_PATH).load()
    for cls in onto.classes():
        label = cls.label
        if label:
            if re.match(r'CL',cls.name):
                labels_all.append(label[0])
                
    labels = []
    for i in clas:
        label = i.label
        if label:
            if label[0] in labels_all:
                if re.match(r'CL',i.name):
                    labels.append(label[0])
    return labels     


def merge_cell_types(ct1, ct2, owl_path=ONTOLOGY_PATH):

    union_labels = set(ct1) | set(ct2)

    onto = get_ontology(owl_path).load()

    def find_nearest_parent(label, current_set):
       
        cls = onto.search_one(label=str(label))
        if cls is None:
            return None

        q = deque([(cls, 0)])
        seen = {cls}

        while q:
            node, dist = q.popleft()
            for parent in node.is_a:
                if not isinstance(parent, ThingClass):
                    continue
                if parent in seen:
                    continue
                seen.add(parent)

                lab = parent.label[0] if parent.label else parent.name
                if lab in current_set:
                    return lab

                q.append((parent, dist + 1))

        return None

    while True:
        new_set = set()
        for lab in union_labels:
            p = find_nearest_parent(lab, union_labels)
            new_set.add(p if p is not None else lab)

        if new_set == union_labels:
            break
        union_labels = new_set

    def resolve(label):
        cls = onto.search_one(label=str(label))
        if cls is None:
            return label

        q = deque([cls])
        seen = {cls}

        while q:
            node = q.popleft()
            lab = node.label[0] if node.label else node.name
            if lab in union_labels:
                return lab
            for parent in node.is_a:
                if isinstance(parent, ThingClass) and parent not in seen:
                    seen.add(parent)
                    q.append(parent)

        return label

    mapped1 = [resolve(x) for x in ct1]
    mapped2 = [resolve(x) for x in ct2]

    return mapped1, mapped2

if __name__ == '__main__':
    pass