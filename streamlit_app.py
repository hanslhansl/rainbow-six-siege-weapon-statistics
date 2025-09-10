import create_weapon_statistics_file, streamlit as st

@st.cache_data
def get_weapons_list():
    return create_weapon_statistics_file.get_weapons_list()

weapons = get_weapons_list()