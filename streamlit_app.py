import create_weapon_statistics_file, streamlit as st, matplotlib.pyplot as plt, pandas as pd, altair as alt, numpy as np, sys
import pandas.core.series
# https://demo-stockpeers.streamlit.app/?ref=streamlit-io-gallery-favorites&stocks=AAPL%2CMSFT%2CGOOGL%2CNVDA%2CAMZN%2CTSLA%2CADP%2CACN%2CABBV%2CAMT%2CAXP%2CAMGN%2CAMD
github_url = "https://github.com/hanslhansl/rainbow-six-siege-weapon-statistics"


@st.cache_data
def get_weapons_dict():
    return create_weapon_statistics_file.get_weapons_dict()

# Load data
weapons = get_weapons_dict()
raw_data = pd.DataFrame({w.name : w.damages for w in weapons.values()}).transpose()

# Streamlit UI
st.set_page_config(
    page_title="r6s weapon stats",
    page_icon=":material/looks_6:", # counter_6 looks_6
    layout="wide",
)
st.markdown("## rainbow six siege weapon statistics")
st.markdown("interactive visualisation of damage data over different distances")
#st.markdown(f"find the [project on github]({github_url})")

"""
### choose stat
"""
with st.container(border=True):
    choice = st.selectbox("Choose one:", ["Option A", "Option B", "Option C", "Option D", "Option E"])
    st.write("You selected:", choice)

    choice = st.radio("Pick one:", ["Option A", "Option B", "Option C", "Option D", "Option E"])
    st.write("You picked:", choice)


"""
### choose weapons
"""
with st.container(border=True):
    cols = st.columns([1, 3])

    # Create rows of buttons
    with cols[0]:
        cols_per_row = 4
        for i in range(0, len(create_weapon_statistics_file.weapon_classes), cols_per_row):
            inner_cols = st.columns(cols_per_row)
            for j, label in enumerate(create_weapon_statistics_file.weapon_classes[i:i+cols_per_row]):
                with inner_cols[j]:
                    if st.button(label):
                        print(f"{label} clicked")

    # Suchfeld und Waffenfilter
    with cols[1]:
        search_query = st.text_input("filter weapons")
        filtered_weapons = [w for w in weapons.values() if search_query.lower() in w.name.lower()]
        selected_weapons = st.multiselect(
            label="select weapon(s)",
            options=filtered_weapons,
            #default=list(st.session_state.selected_weapons),
            format_func=lambda w: w.name
            )

data = {w.name : w.damages for w in selected_weapons}

"""
### plot
"""
right_cell = st.container(border=True)
with right_cell:
    if len(data):
        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # insert average
        if len(data) > 1:
            df.insert(0, "average", df[data.keys()].mean(axis=1))

        # transform to long form
        df["distance"] = df.index
        df = df.melt(id_vars="distance", var_name="weapon", value_name="damage")

        # calculate y axis range with padding
        y_min = df["damage"].min() if selected_weapons else 0
        y_max = df["damage"].max() if selected_weapons else 100
        padding = (y_max - y_min) * 0.1

        # plot altair chart
        #st.line_chart(df, height=700)
        st.altair_chart(
            alt.Chart(df).mark_point().mark_line().encode(
                x="distance",
                y=alt.Y("damage", scale=alt.Scale(domain=[y_min - padding, y_max + padding])),
                color=alt.Color("weapon", sort=["average"])
            ).properties(
                height=600
            )
        )
    else:
        st.info("select some weapons to plot")


"""
### raw data
"""

df = raw_data.reset_index().rename(columns={"index": "weapon"})
def color_survived(val):
    color = 'green' if val else 'red'
    return f'background-color: {color}'
def column_styler(s : pandas.core.series.Series):
    weapon_string = s["weapon"]
    weapon = weapons[weapon_string]
    
    ret = []
    for index, value in s.items():
        if index == "weapon":
            ret.append(f"background-color: #{weapon.pd_name_color()}")
        else:
            ret.append(f"background-color: #{weapon.pd_damage_drop_off_color(index)}")

    return ret


df = (
    df.style
    #.map(color_survived, subset=[2])
    #.map(color_survived, subset=["weapons"])
    .apply(column_styler, axis=1)
#df=df.style.highlight_max(color="lightgreen", axis=0)
)

if 0:
    config = {i+2 : st.column_config.Column(width=1) for i in range(0, len(create_weapon_statistics_file.Weapon.distances))}
    config[1] = st.column_config.Column(pinned=True, width=200)

    st.dataframe(
        df,
        height=700,
        column_config=config,
        hide_index=True
    )
else:
    st.table(df)