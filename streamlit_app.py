import create_weapon_statistics_file as cwsf, streamlit as st, matplotlib.pyplot as plt, pandas as pd, altair as alt, numpy as np, sys
import pandas.core.series
# https://demo-stockpeers.streamlit.app/?ref=streamlit-io-gallery-favorites&stocks=AAPL%2CMSFT%2CGOOGL%2CNVDA%2CAMZN%2CTSLA%2CADP%2CACN%2CABBV%2CAMT%2CAXP%2CAMGN%2CAMD

@st.cache_data
def get_weapons_dict():
    return cwsf.get_weapons_dict()
@st.cache_data
def get_weapons():
    return cwsf.Weapons()

# Load data
weapons = get_weapons()
#raw_data = pd.DataFrame({w.name : w.damages for w in weapons.values()}).transpose()

# Streamlit UI
st.set_page_config(
    page_title="r6s weapon stats",
    page_icon=":material/looks_6:", # counter_6 looks_6
    layout="wide",
)
st.markdown("## rainbow six siege weapon statistics")
st.markdown("interactive visualisation of damage data over different distances")
#st.markdown(f"find the [project on github]({cwsf.github_url})")

"""
### choose stat
"""
with st.container(border=True):
    selected_stat = st.selectbox(
        "choose a stat:",
        cwsf.stats,
        format_func=lambda stat: stat.name if stat.name == stat.short_name else f"{stat.short_name} - {stat.name}"
        )
    
    selected_illustration = st.selectbox(
        "choose a coloring scheme:",
        cwsf.stat_illustrations,
        format_func=lambda x: x.__doc__
        )


"""
### choose weapons
"""
with st.container(border=True):
    cols = st.columns([1, 3])

    # Create rows of buttons
    with cols[0]:
        cols_per_row = 4
        for i in range(0, len(cwsf.Weapon.classes), cols_per_row):
            inner_cols = st.columns(cols_per_row)
            for j, label in enumerate(cwsf.Weapon.classes[i:i+cols_per_row]):
                with inner_cols[j]:
                    if st.button(label):
                        print(f"{label} clicked")

    # Suchfeld und Waffenfilter
    with cols[1]:
        search_query = st.text_input("filter weapons")
        filtered_weapons = [name for name in weapons.weapons if search_query.lower() in name.lower()]
        selected_weapons = st.multiselect(
            label="select weapon(s)",
            options=filtered_weapons,
            #default=list(st.session_state.selected_weapons),
            )

if len(selected_weapons):
    selected_weapons = selected_weapons
else:
    selected_weapons = list(weapons.weapons)

"""
### plot
"""
with st.container(border=True):
    if False:#len(selected_weapons):
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

def column_styler(s : pandas.core.series.Series):
    weapon = weapons[s["weapon"]]
    
    ret = []
    for index, value in s.items():
        if index == "weapon":
            ret.append(f"background-color: {weapon.name_color.to_css()}")
        else:
            ret.append(f"background-color: {selected_illustration(weapon, index).to_css()}")

    return ret

df = (
    weapons.damages()[selected_weapons]
    .transpose()
    .reset_index()
    .rename(columns={"index": "weapon"})
    #.style.apply(column_styler, axis=1)
    )

#df.to_excel("out.xlsx")

if False:
    config = {i+2 : st.column_config.Column(width=1) for i in range(0, len(cwsf.Weapon.distances))}
    config[1] = st.column_config.Column(pinned=True, width=200)

    st.dataframe(
        df,
        height=700,
        column_config=config,
        hide_index=True
    )
else:
    st.table(df)