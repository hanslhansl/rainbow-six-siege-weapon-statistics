import create_weapon_statistics_file, streamlit as st, matplotlib.pyplot as plt, pandas as pd

# https://demo-stockpeers.streamlit.app/?ref=streamlit-io-gallery-favorites&stocks=AAPL%2CMSFT%2CGOOGL%2CNVDA%2CAMZN%2CTSLA%2CADP%2CACN%2CABBV%2CAMT%2CAXP%2CAMGN%2CAMD
github_url = "https://github.com/hanslhansl/rainbow-six-siege-weapon-statistics"

@st.cache_data
def get_weapons_list():
    return create_weapon_statistics_file.get_weapons_list()

# Calculate average damage per distance across selected weapons
def calculate_average_damage(selected_weapons : create_weapon_statistics_file.Weapon):
    num_weapons = len(selected_weapons)
    return [sum(w.damages[i] for w in selected_weapons) / num_weapons for i in range(len(create_weapon_statistics_file.Weapon.distances))]
    

# Load data
weapons = get_weapons_list()
print(weapons)
raw_data = pd.DataFrame({w.name : w.damages for w in weapons}).transpose()

# Initialize session state
if "selected_weapons" not in st.session_state:
    temp : set[create_weapon_statistics_file.Weapon] = set()
    st.session_state.selected_weapons = temp

# Streamlit UI
st.set_page_config(
    page_title="r6s weapon stats",
    page_icon=":material/looks_6:", # counter_6 looks_6
    layout="wide",
)
st.title("rainbow six siege weapon statistics")
st.markdown("interactive visualisation of damage data over different distances")
st.markdown(f"find [project on github]({github_url})")

cols = st.columns([1, 3])

top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:

    #Button to add subset
    if st.button("ARs"):
        # Merge current selection with subset, avoiding duplicates
        st.session_state.selected_weapons.update(w for w in weapons if w.class_ == "AR")
        #st.session_state.selected_weapons = st.session_state.selected_weapons + {w for w in weapons if w.class_ == "AR"}

    # Suchfeld und Waffenfilter
    search_query = st.text_input("🔍 search weapon")
    filtered_weapons = [w for w in weapons if search_query.lower() in w.name.lower()]
    try:
        selected_weapons = st.multiselect(
            label="select weapon",
            options=filtered_weapons,
            default=list(st.session_state.selected_weapons),
            format_func=lambda w: w.name
            )
    except st.errors.StreamlitAPIException:
        print("filtered_weapons:", filtered_weapons)
        print("st.session_state.selected_weapons:", st.session_state.selected_weapons)
        raise

    # Durchschnittsanzeige
    show_average = st.checkbox("show average")


right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)

with right_cell:
# Plot erstellen
    if selected_weapons:
        fig, ax = plt.subplots()
        for weapon in selected_weapons:
            ax.plot(create_weapon_statistics_file.Weapon.distances, weapon.damages, label=weapon.name)
        
        if show_average:
            avg_damage = calculate_average_damage(selected_weapons)
            ax.plot(create_weapon_statistics_file.Weapon.distances, avg_damage, label="Average", linestyle='--', color='black')

        ax.set_xlabel("distance (meter)")
        ax.set_ylabel("damage per bullet")
        ax.set_title("damage over distance")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("selet at least one weapon")


"""
## raw data
"""

st.dataframe(raw_data, height=700)