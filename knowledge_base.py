# %%
import os
import json
import pickle
import streamlit as st
from random import randrange

from haystack import Pipeline
from haystack import Document
from haystack.core.component import Component
from haystack import component
from haystack.document_stores.in_memory import InMemoryDocumentStore
# from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# from haystack.components.converters import PDFMinerToDocument
from haystack.components.converters import PyPDFToDocument, TextFileToDocument  #  After minimal Testing, PyPDFToDocument appears to be working better for our documents 
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
# from haystack.components.preprocessors import RecursiveDocumentSplitter

import contextualiser

#%%
CHUNK_SIZE = 250

# path_choma_document_store = os.path.join("data", "chroma", "documents_KB")
data_folder = "data"
docs_folder = os.path.join(data_folder, "knowledge_base_documents")

docs_names = os.listdir(docs_folder)
docs_paths = [os.path.join(docs_folder, d) for d in docs_names]

pdf_kb = False
sc_kb = True
comb_kb = False

# %%
""" Testing
converter = PyPDFToDocument()
docs = converter.run(sources=docs_paths[4:6])['documents']

cleaner = DocumentCleaner()
docs_c = cleaner.run(docs)['documents']

# splitter2 = RecursiveDocumentSplitter(split_length=CHUNK_SIZE)
# splitter2.warm_up()
# docs_s = splitter2.run(docs_c)['documents']

splitter = DocumentSplitter(split_by = "word", split_length = CHUNK_SIZE, split_overlap = 25)
docs_s = splitter.run(docs_c)

"""

#%%
def preprocess_pdfs(document_store):
    print("pdf preprocessing beginning")

    openai_key = st.secrets["API_keys"]["openai"]

    document_splitter = DocumentSplitter(split_by="word", split_length=CHUNK_SIZE, split_overlap=50)
    document_contextualiser = contextualiser.ContextualTextPreProcessor(chunk_size=CHUNK_SIZE, api_key=openai_key)
    document_embedder = SentenceTransformersDocumentEmbedder()
    # TODO - Add bm25 embedding pdf files
    # TODO - It cannot read and convert to Documents, two of the pdfs, cryptography>=3.1 is required for AES algorithm
    document_writer = DocumentWriter(document_store)
    
    pdf_preprocessing_pipeline = Pipeline()
    pdf_preprocessing_pipeline.add_component(instance=PyPDFToDocument(), name="pdf_converter")
    pdf_preprocessing_pipeline.add_component(instance=DocumentCleaner(), name="document_cleaner")
    pdf_preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    pdf_preprocessing_pipeline.add_component(instance=document_contextualiser, name="document_contextualiser")
    pdf_preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    pdf_preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

    pdf_preprocessing_pipeline.connect("pdf_converter", "document_cleaner")
    pdf_preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    pdf_preprocessing_pipeline.connect("document_cleaner", "document_contextualiser.og_docs")
    pdf_preprocessing_pipeline.connect("document_splitter", "document_contextualiser.documents")
    pdf_preprocessing_pipeline.connect("document_contextualiser", "document_embedder")
    pdf_preprocessing_pipeline.connect("document_embedder", "document_writer")
    pdf_preprocessing_pipeline.run({"pdf_converter": {"sources": docs_paths}})
    print("pdf preprocessing pipeline finished")

    return document_store


def input_scenario_to_string_variation_2(s):
    string = f"Student Profile:\n{json.dumps(s['student_profile'])}\n\nSituation:\n{s['situation']}\n\nAction Taken:\n{s['action']}"
    return string

def convert_all_json_to_text():
    path_scenario_folder = os.path.join("data", "new_synthetic_scenarios")
    paths_scenarios = [os.path.join(path_scenario_folder, s) for s in os.listdir(path_scenario_folder)]
    scenarios = []
    for path in paths_scenarios:
        with open(path, "r") as f:
            s = json.load(f)
            s['file_path'] = path
        
        meta = {
                    "file": s['file_path'],
                    "grade": s['grade'],
                    "effect": s['effect'],
                    "topic": s['topic'],
                    "len": s['len']
                }   
        text = input_scenario_to_string_variation_2(s)

        scenarios.append(Document(content = text, meta=meta))
    return scenarios


def preprocess_scenarios(document_store):

    # TODO Decide what to do with the metadata 
    print("scenarios preprocessing beginning")
    scenarios_txt = convert_all_json_to_text()

    scenario_preprocessing_pipeline = Pipeline()
    json_embedder = SentenceTransformersDocumentEmbedder()
    # TODO - Add bm25 embedding txt files
    document_writer_json = DocumentWriter(document_store)

    scenario_preprocessing_pipeline.add_component(instance=json_embedder, name="json_embedder")
    scenario_preprocessing_pipeline.add_component(instance=document_writer_json, name="document_writer_json")

    scenario_preprocessing_pipeline.connect("json_embedder", "document_writer_json")

    scenario_preprocessing_pipeline.run({"json_embedder": {"documents": scenarios_txt}})
    print("json preprocessing pipeline finished")
    
    return document_store

def get_rand_scenario_high_grade():
    path_scenario_folder = os.path.join("data", "new_synthetic_scenarios")
    paths_scenarios = [os.path.join(path_scenario_folder, s) for s in os.listdir(path_scenario_folder)]
    scenarios_grade_five = []
    for path in paths_scenarios:
        with open(path, "r") as f:
            s = json.load(f)
        if s['grade']==5:
            scenarios_grade_five.append(path)
    scenario_path = scenarios_grade_five[randrange(0, len(scenarios_grade_five))]
    return scenario_path

# %% Combining pdf documents with txt scenarios 

def create_pdf_kb():
    document_store_pdf = preprocess_pdfs(InMemoryDocumentStore(embedding_similarity_function="cosine"))
    return document_store_pdf

def create_scenario_kb():
    document_store_scenario = preprocess_scenarios(InMemoryDocumentStore(embedding_similarity_function="cosine"))
    return document_store_scenario

def create_combined_kb():

    kb_pdfs_pkl_file_path = os.path.join(data_folder, "doc_store_pdfs.pkl")
    if os.path.exists(kb_pdfs_pkl_file_path):
        doc_store_pdfs = InMemoryDocumentStore.load_from_disk(kb_pdfs_pkl_file_path)
    else:
        doc_store_pdfs = preprocess_pdfs(InMemoryDocumentStore(embedding_similarity_function="cosine"))

    document_store_combined = preprocess_scenarios(doc_store_pdfs)
    return document_store_combined

# %% 
if __name__ == "__main__":
    if pdf_kb:
        document_store_pdf = create_pdf_kb()
        document_store_pdf.save_to_disk("data/doc_store_pdfs.pkl")

    if sc_kb:
        document_store_scenario = create_scenario_kb()
        document_store_scenario.save_to_disk("data/doc_store_scenarios.pkl")
        
    if comb_kb:
        document_store_combined = create_combined_kb() 
        document_store_pdf.save_to_disk("data/doc_store_combined.pkl")