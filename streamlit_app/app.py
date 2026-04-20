import streamlit as st


st.set_page_config(page_title="Estate Planner Chat", page_icon=":scroll:", layout="wide")

main_page = st.Page("pages/1_main.py", title="Chat", default=True)
documents_page = st.Page("pages/2_documents.py", title="Documents")
people_page = st.Page("pages/3_people.py", title="Persons")

navigation = st.navigation([main_page, people_page, documents_page])
navigation.run()
