import streamlit as st
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

openai_key = st.secrets["API_keys"]["openai"]
client = openai.OpenAI(api_key = openai_key)

@st.cache_resource
def start_knowledge_base():
    path_document_store = os.path.join("data", "doc_store_pdfs_sent.pkl")
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
    query_pipeline.warm_up()
    return doc_store_pdf, query_pipeline

doc_store, pipeline = start_knowledge_base()

def query_knowledge_base(query_text, n=5):
    res = pipeline.run({"text_embedder": {"text": query_text}, "retriever": {"top_k": n}})
    return res['retriever']['documents']

def format_chunks(documents):
    # chunks = [d.content for d in result['retriever']['documents'] if d.score>0.2]
    chunks_all_info = [f"""Content: {d.content}, Filepath: {d.meta['file_path']}, page number: 
                        {d.meta['page_number']}, URL: {d.meta['url']}, Score: {d.score}""" for d in documents] # if d.score>0.2]
    # meta_chunks = [[d.meta['file_path'], d.meta['page_number'], d.meta['url'], d.score] for d in result['retriever']['documents'] if d.score>0.2]
    chunks_str = "\n\n".join(chunks_all_info)
    return chunks_str

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
    st.title("Virtual Assistant for SEND")

    st.markdown(
        """
        This app helps teachers working with autistic students by:
        - **Rating an action** given a student profile and situation.
        - **Suggesting an action** for a specific student profile and situation.
        - Training tool for **Self Assessment** for teachers
        """
    )

    # Tabs for the three functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Rate an Action", "Suggest an Action", "Self Assessment - Training", "Conversational"])

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
                    # TODO @st.cache_resourse 
                    
                    query = student_profile + " " + situation
                    result = query_knowledge_base(query)
                    chunks_prompt = format_chunks(result)
                    
                    suggest_action_kb_prompt = f"""
                    You are a helpful assistant that helps resolving problematic situations involving student with special educational needs.
                    The profile of the student is:
                    {student_profile}.
                    The situation that happened with the student is:
                    {situation}.
                    PDF document chunks:
                    {chunks_prompt}
                    Taking into account the student profile, the situation and only the "Content" information from the chunks, 
                    suggest what would be the best and most effective action in such situation in a short paragraph with up to 3 step.

                    Using the content and metadata from all the chunks you found usefull and used to generate the answer, and
                    output the result in the format:
                    
                    File name: 
                    \n
                    Page:
                    \n
                    URL:
                    \n
                    Score: 
                    """

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": suggest_action_kb_prompt}
                        ]
                    )
                    suggested_action = response.choices[0].message.content

                    st.success(f"Suggested Action: {suggested_action}")
              
#                     for id_chunk, chunk in enumerate(chunks):
#                         st.write(f"""Chunk: {chunk} \n
# File Name: {meta_chunks[id_chunk][0]} \n
# Page: {meta_chunks[id_chunk][1]} \n
# URL: {meta_chunks[id_chunk][2]} \n
# Score: {meta_chunks[id_chunk][3]} \n
# ======================================================= \n""")                     

            else:
                st.warning("Please fill in the student profile and situation before proceeding.")

    with tab3:
            st.subheader("Self Assessment - Training")

        
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
    
    
    
    
    with tab4:
            
        SYSTEM_MESSAGE_HYBRID = {"role": "system", "content": """
        You are a helpful assistant that helps resolving problematic situations involving student with special educational needs. 
        Your tasks are to either help someone by suggesting proper actions in such a problematic scenario or rating an action described by the user by giving feedback and suggestions for improvement.
        Before suggesting or rating an action you have to gather enough Information about the scenario and the participants!
        Information you always want to know include the persons included in the scenario and their profile. You want to know how old they are their gender and if they have special needs their diagnosis
        For the scenario you want to know what initiated this problematic situation and what was the setting (place, time ...) 
        If you feel like you gathered enough information to suggest or rate and action start your message with '__SUGGESTION__' or '__RATING__' respectively.
        """}

        SYSTEM_MESSAGE_SUGGEST = {"role": "system", "content": """
        You are a helpful assistant that helps resolving problematic situations involving student with special educational needs. 
        Your tasks are to help someone by suggesting proper actions in such a problematic scenario to help the use solve or better the situation.
        Before suggesting or rating an action you have to gather enough Information about the scenario and the participants in a conversational manner!
        Information you always want to know include the persons included in the scenario and their profile. You want to know how old they are their gender and if they have special needs their diagnosis.
        For the scenario you want to know what initiated this problematic situation or if there were any triggers and what was the setting (place, time ...) 
        If you feel like you gathered enough information to suggest an action start your message with the keyword '__SUGGESTION__' so that the system can properly format the following suggestion.
        """}

        SYSTEM_MESSAGE_TOOL = {"role": "system", "content": """
        You are a helpful assistant that helps resolving problematic situations involving student with special educational needs. 
        Your tasks are to help someone by suggesting proper actions in such a problematic scenario to help the use solve or better the situation.
        Before suggesting or rating an action you have to gather enough Information about the scenario and the participants in a conversational manner!
        Information you always want to know include the persons included in the scenario and their profile. You want to know how old they are their gender and if they have special needs their diagnosis.
        For the scenario you want to know what initiated this problematic situation or if there were any triggers and what was the setting (place, time ...) 
        If you feel like you gathered enough information to suggest an action, before suggesting anything i expect you return a tool call calling the query_knowledge_base function with a small summary as and input!! 
        You will get relevant chunks as an input to then make a suggestion.        
        I demand that you always in every chat after gathering sufficient information about the situation and before making a suggestion you make this tool call so that your answer can be enhanced and the user can be provided with source to believe the application.
        """}

        TOOLS = [{
            "type": "function",
            "function": {
                "name": "query_knowledge_base",
                "description": "Semantically query the systems rich Knowledge Base of Documents for Guidelines on Handling and Education people with special educational needs. Intended to be used with a summary of a problematic situation regarding special needs people. It will return the top 5 most relevant chunks of the Knowledge Base regarding the scenario",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "A summary of the situation used to semantically query the Knowledge base"
                        }
                    },
                    "required": [
                        "query_text"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]

        WELCOME_MESSAGE = {"role": "assistant", "content": "Hello, how can i help you today?"}
        SYSTEM_MESSAGE = SYSTEM_MESSAGE_TOOL # Choose version

        if "messages" not in st.session_state:
            st.session_state.messages = [SYSTEM_MESSAGE, WELCOME_MESSAGE]

        # Reset button to clear chat
        if st.button("🔄 Start New Chat"):
            st.session_state.messages = [SYSTEM_MESSAGE, WELCOME_MESSAGE]
            st.rerun()


        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.messages:
                if type(msg) == dict:  
                    role = msg.get('role', 'else')
                    if role == "user" or role == "assistant":
                        with st.chat_message(role):
                            st.markdown(msg["content"])


        # Add an empty container below messages (pushing input box to the bottom)
        st.empty()

        # Input box always at the bottom
        user_input = st.chat_input("Type your message here...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

            with st.spinner("Thinking..."):
                response = client.chat.completions.create(model="o3-mini", messages=st.session_state.messages, tools=TOOLS)
                response_message = response.choices[0].message.content
                
                print(response)
                if response_message is not None:                   
                    st.session_state.messages.append({"role": "assistant", "content": response_message})
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(response_message)


                else: 
                    print("\n\nTOOL CALLING\n")
                    st.session_state.messages.append(response.choices[0].message)

                    tool_call = response.choices[0].message.tool_calls[0]
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_input = tool_args['query_text']

                    documents = query_knowledge_base(tool_input)
                    formated_chunks = format_chunks(documents)

                    st.session_state.messages.append({                               # append result message
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": formated_chunks
                    })

                    system_context_message = {"role": "system", "content": f"""
                    You the Assistant just made the tool call query_knowledge_base():
                    Taking into account the persons profile, the situation and only the "Content" information from the chunks, 
                    suggest what would be the best and most effective action in such situation in a short paragraph with up to 3 step.

                    Also cite chunks from documents u found usefull. Append you suggestion by outputing citations in the format:

                    File name: 
                    \n
                    Page:
                    \n
                    URL:
                    """}

                    st.session_state.messages.append(system_context_message)

                    response = client.chat.completions.create(model="o3-mini", messages=st.session_state.messages, tools=TOOLS)
                    response_message = response.choices[0].message.content
                    print(response)

                    if response_message is not None:
                        st.session_state.messages.append({"role": "assistant", "content": response_message})

                        with chat_container:
                            with st.chat_message("assistant"):
                                st.markdown(response_message)
                    else:
                        print("\n\n OH Oh! Double Tool Call\n")
    # Footer
    st.markdown("---")
    st.markdown("Developed for showcasing purposes only - No real Scenarios used")
