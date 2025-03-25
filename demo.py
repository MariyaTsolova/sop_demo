import streamlit as st
from random import randrange
import pandas as pd
import json
import openai
import os

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.embedders import SentenceTransformersTextEmbedder

import knowledge_base

import yaml
import hashlib

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Action Recommender Demo",
    layout="centered",
    # initial_sidebar_state="collapsed"
)

CONFIG_FILE = "config_credentials.yaml"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Load credentials from the config file
def load_users():
    if not os.path.exists(CONFIG_FILE):
        st.error("Configuration file not found. Please generate it using the setup script.")
        return None

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["credentials"]["users"]

# Verify user credentials using hashed passwords
def authenticate_user(username, password):
    users = load_users()
    if users is None:
        return False

    hashed_password = hash_password(password)
    return username in users and users[username]["password"] == hashed_password

# Streamlit authentication UI
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if authenticate_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.sidebar.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials. Please try again.")

# Logout function
def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Show login page if not authenticated
if not st.session_state["authenticated"]:
    login()
else:
    if st.sidebar.button("Logout"):
        logout()  
        st.rerun()  


# Main Web App
    st.title("Action Recommender for Teachers")

    st.markdown(
        """
        This app helps teachers working with autistic students by:
        - **Rating an action** given a student profile and situation.
        - **Suggesting an action** for a specific student profile and situation.
        - Training platform for **Self Assessing tool** for teachers
        """
    )

    # Tabs for the three functionalities
    tab1, tab2, tab3 = st.tabs(["Rate an Action", "Suggest an Action", "Self Assessment"])

    # First Tab: Rate an Action
    with tab1:
        st.subheader("Rate an Action")

        if "text_student_profile" not in st.session_state:
            st.session_state.text_student_profile = ""
        if "text_situation" not in st.session_state:
            st.session_state.text_situation = ""
        if "text_action" not in st.session_state:
            st.session_state.text_action = ""

        clear_tab1 = st.button('Clear and start a new situation', key="tab1")
        if clear_tab1:
            st.session_state.text_student_profile = ""
            st.session_state.text_situation = ""
            st.session_state.text_action = ""

        student_profile = st.text_area("Student Profile:", placeholder="Describe the student's profile...", key = "text_student_profile")
        situation = st.text_area("Situation:", placeholder="Describe the action to be rated", key = "text_situation")
        action = st.text_area("Action:", placeholder="Describe the current situation", key = "text_action")

        # use_knowledge_base = st.checkbox("Use Knowledge Base")

        rate_action_prompt = f"""
        You are a helpful assistant that helps resolving problematic situations involving student with special educational needs.
        The profile of the student is:
        {student_profile}.
        The situation that happened with the student is:
        {situation}.
        The action that was taken to resolve the situation is:
        {action}.
        Rate the action from 1 to 5, 1 being very ineffective and 5 being very effective
        and say with 1-2 sentences why you give such rate.
        Format the response as follows:
        Rate: [a number from 1 to 5]
        Comment: [the explanation]        
        """

        if st.button("Rate Action"):
            
            if st.session_state.text_student_profile and situation and action:
                openai_key = st.secrets["API_keys"]["openai"]
                client = openai.OpenAI(api_key = openai_key)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": rate_action_prompt}
                    ]
                ) 

                st.write(response.choices[0].message.content)
            else:
                st.warning("Please fill in all fields before proceeding.")


        st.markdown("""
        The rating is from 1 to 5:
        - 1 - bad reaction
        - 2 - ineffective reaction
        - 3 - slightly effective reaction
        - 4 - effective reaction
        - 5 - very effective reaction 
        """)

    # Second Tab: Suggest an Action
    with tab2:
        st.subheader("Suggest an Action")

        if "text_student_profile_tab2" not in st.session_state:
            st.session_state.text_student_profile_tab2 = ""
        if "text_situation_tab2" not in st.session_state:
            st.session_state.text_situation_tab2 = ""

        clear_tab2 = st.button('Clear and start a new situation', key="tab2")
        if clear_tab2:
            st.session_state.text_student_profile_tab2 = ""
            st.session_state.text_situation_tab2 = ""


        student_profile = st.text_area("Student Profile:", placeholder="Describe the student's profile...", key = "text_student_profile_tab2")
        situation = st.text_area("Situation:", placeholder="Describe the action to be rated...", key = "text_situation_tab2")
        use_knowledge_base = st.checkbox("Use Knowledge Base", key="suggest_kb")

        suggest_action_prompt = f"""
        You are a helpful assistant that helps resolving problematic situations involving student with special educational needs.
        The profile of the student is:
        {student_profile}.
        The situation that happened with the student is:
        {situation}.
        Suggest what would be the best and most effective action in such situation in a short paragraph with up to 3 steps taking into accout the student's profile.
        """

        if use_knowledge_base:
            service = 2
        else: 
            service = 1

        if st.button("Suggest Action"):
            
            if student_profile and situation:
                if service == 1:
                    # Call your action suggestion function
                    openai_key = st.secrets["API_keys"]["openai"]
                    client = openai.OpenAI(api_key = openai_key)

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": suggest_action_prompt}
                        ]
                    )
                    suggested_action = response.choices[0].message.content

                    st.success(f"Suggested Action: {suggested_action}")

                if service == 2:
                    path_document_store = os.path.join("data", "doc_store_pdfs.pkl")
                    doc_store_pdf = InMemoryDocumentStore.load_from_disk(path_document_store)                   
                    
                    # # BM25 Retriever
                    # retriever = InMemoryBM25Retriever(document_store=doc_store_pdf)
                    # pipeline = Pipeline()
                    # pipeline.add_component(instance=retriever, name="retriever")
                    # result = pipeline.run(data={"retriever": {"query":"Age: 10, Gender: female, Diagnosis: ADHD. Situation: The kid fell from the chair and hurt his head."}})               
                    # result['retriever']['documents'][0].content

                    # Embedding Retriever
                    query_pipeline = Pipeline()
                    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
                    query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=doc_store_pdf))
                    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

                    query = student_profile + " " + situation
                    result = query_pipeline.run({"text_embedder": {"text": query}})

                    chunks = [d.content for d in result['retriever']['documents'] if d.score>0.4]
                    chunks = "\n\n".join(chunks)

                    suggest_action_kb_prompt = f"""
                    You are a helpful assistant that helps resolving problematic situations involving student with special educational needs.
                    The profile of the student is:
                    {student_profile}.
                    The situation that happened with the student is:
                    {situation}.
                    PDF document chunks:
                    {chunks}
                    Taking into account the student profile, the situation and the information from the chunks, 
                    suggest what would be the best and most effective action in such situation in a short paragraph with up to 3 step.
                    """

                    openai_key = st.secrets["API_keys"]["openai"]
                    client = openai.OpenAI(api_key = openai_key)

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": suggest_action_kb_prompt}
                        ]
                    )
                    suggested_action = response.choices[0].message.content

                    st.success(f"Suggested Action: {suggested_action}")
                    
            else:
                st.warning("Please fill in the student profile and situation before proceeding.")

    with tab3:
            st.subheader("Self Assess")


        
            if "data" not in st.session_state:
                scenario_path = knowledge_base.get_rand_scenario_high_grade()
                with open(scenario_path, 'r') as file:
                    st.session_state.data = json.load(file)
                st.session_state.text_reaction_test = ""  
                st.session_state.text_reaction_true = ""
        
        
            if "text_situation_gen" not in st.session_state:
                st.session_state.text_situation_gen = f"""Age of student: {st.session_state.data['student_profile']['age']}
        Gender: {st.session_state.data['student_profile']['gender']}
        Conditions: {', '.join(st.session_state.data['student_profile']["diagnosis"])}
        \nSituation: {st.session_state.data['situation']}"""
        
            
            if st.button("New Scenario"):
                scenario_path = knowledge_base.get_rand_scenario_high_grade()
                with open(scenario_path, 'r') as file:
                    st.session_state.data = json.load(file)
        
                # Reset responses
                st.session_state.text_reaction_test = ""
                st.session_state.text_reaction_true = ""
        
                # Update the displayed situation text
                st.session_state.text_situation_gen = f"""Age of student: {st.session_state.data['student_profile']['age']}
        Gender: {st.session_state.data['student_profile']['gender']}
        Conditions: {', '.join(st.session_state.data['student_profile']["diagnosis"])}
        \nSituation: {st.session_state.data['situation']}"""
        
        
            # text_area_sc = st.text_area("Situation",  
            #                             st.session_state.text_situation_gen,  
            #                             height=170,  
            #                             label_visibility='collapsed',
            #                              disabled=True)



            st.markdown(f"""
                    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f8f9fa; color: black; font-size: 16px;">
                        {st.session_state.text_situation_gen}
                    </div>
                """, unsafe_allow_html=True)
        
            text_area_react_test = st.text_area("What would you do?",  
                                                height=150,  
                                                key="text_reaction_test")  
        
            
            if st.button("Submit"):
                st.session_state.text_reaction_true = st.session_state.data['action']
        
            



           

            # Non-editable text box
            # text_area_react_true = st.text_area("Reaction",  
            #                                     height=170,  
            #                                     label_visibility='collapsed',  
            #                                     key="text_reaction_true",
            #                                     disabled=True)

            st.markdown(f"""
                    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f8f9fa; color: black; font-size: 16px;">
                        {st.session_state.text_reaction_true}
                    </div>
                """, unsafe_allow_html=True)
                                        



############################################################

    #         text_area_react_true = st.text_area("Reaction",  
    #                                             height=170,  
    #                                             label_visibility='collapsed',  
    #                                             key="text_reaction_true",
    #                                              disabled=True)  
    # Footer
    st.markdown("---")
    st.markdown("Developed for showcasing purposes only - No real Scenarios used")



