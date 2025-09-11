import create_weapon_statistics_file, streamlit as st, matplotlib.pyplot as plt, pandas as pd

# https://demo-stockpeers.streamlit.app/?ref=streamlit-io-gallery-favorites&stocks=AAPL%2CMSFT%2CGOOGL%2CNVDA%2CAMZN%2CTSLA%2CADP%2CACN%2CABBV%2CAMT%2CAXP%2CAMGN%2CAMD

@st.cache_data
def get_weapons_list():
    return create_weapon_statistics_file.get_weapons_list()

# Calculate average damage per distance across selected weapons
def calculate_average_damage(selected_weapons : create_weapon_statistics_file.Weapon):
    num_weapons = len(selected_weapons)
    return (sum(w.damages[i] for w in selected_weapons) / num_weapons for i in range(len(create_weapon_statistics_file.Weapon.distances)))
    

# Load data
weapons = get_weapons_list()

# Streamlit UI
st.set_page_config(
    page_title="R6S Weapon Statistics",
    page_icon=":material/looks_6:", # counter_6 looks_6
    layout="wide",
)
st.title("Rainbow Six Siege Weapon Statistics")
st.markdown("Interaktive Visualisierung der Schadensdaten pro Kugel über verschiedene Distanzen.")

cols = st.columns([1, 3])

top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    # Suchfeld und Waffenfilter
    search_query = st.text_input("🔍 Waffe suchen")
    filtered_weapons = [w for w in weapons if search_query.lower() in w.name.lower()]
    selected_weapons = st.multiselect(
        label="Waffen auswählen",
        options=filtered_weapons,
        #default=filtered_weapons,
        format_func=lambda w: w.name
        )

    # Durchschnittsanzeige
    show_average = st.checkbox("Durchschnitt über ausgewählte Waffen anzeigen")

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

        ax.set_xlabel("Distanz (Meter)")
        ax.set_ylabel("Schaden pro Kugel")
        ax.set_title("Schadensverlauf über Distanz")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Bitte mindestens eine Waffe auswählen.")


"""
## Raw data
"""

df = pd.DataFrame({w.name : w.damages for w in weapons}).transpose()
df