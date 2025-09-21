import create_weapon_statistics_file as cwsf, streamlit as st, matplotlib.pyplot as plt, pandas as pd, altair as alt, numpy as np, sys
import pandas.core.series
# https://demo-stockpeers.streamlit.app/?ref=streamlit-io-gallery-favorites&stocks=AAPL%2CMSFT%2CGOOGL%2CNVDA%2CAMZN%2CTSLA%2CADP%2CACN%2CABBV%2CAMT%2CAXP%2CAMGN%2CAMD

import streamlit as st


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
        format_func=lambda stat: stat.display_name
        )

    extended_barrel_difference = st.checkbox("calculate the difference between the weapon with and without extended barrel", False)

    additional_parameter = st.pills(
        label=f"choose {selected_stat.additional_parameter_name} level",
        options=selected_stat.additional_parameters,
        default=selected_stat.additional_parameters[0],
        format_func=lambda x: selected_stat.additional_parameters_descriptions[selected_stat.additional_parameters.index(x)],
        label_visibility="collapsed",
        ) if selected_stat.additional_parameters else None

"""
### choose weapons
"""
with st.container(border=True):

    def add_class_to_selection(class_):
        st.session_state.selected_weapons_multiselect = list(dict.fromkeys(
            st.session_state.selected_weapons_multiselect
            + [name for name, w in weapons.base_weapons.items() if w.class_==class_]
            ))

    selected_weapons = st.multiselect(
        label="select weapon(s)",
        options=weapons.base_weapons,
        key="selected_weapons_multiselect"
        )

    include_eb = st.checkbox(
        label="include extended barrel stats",
        value=True
        )
    
    # Create rows of buttons
    for inner_col, class_ in zip(st.columns(len(cwsf.Weapon.classes)), cwsf.Weapon.classes):
        with inner_col:
            st.button(class_, on_click=lambda c=class_: add_class_to_selection(c))


    if len(selected_weapons):
        selected_weapons = selected_weapons
    else:
        selected_weapons = list(weapons.base_weapons)

    if include_eb:
        l = lambda w: [w.name, w.extended_barrel_weapon.name] if w.extended_barrel_weapon else [w.name]
        selected_weapons = [x for n in selected_weapons for x in l(weapons.base_weapons[n])]

_ = """
### plot

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


"""
### data
"""
with st.container(border=True):
    selected_illustration = st.selectbox(
            "choose a coloring scheme:",
            cwsf.stat_illustrations,
            format_func=lambda x: x.__name__.replace("_", " ")
            )

st.write(selected_illustration.__doc__.format(stat=selected_stat.short_name))

target = selected_stat.stat_method(weapons, additional_parameter).loc[selected_weapons]

source = None
if extended_barrel_difference:
    has_or_is_eb = weapons.filter(target, lambda w: w.is_extended_barrel or w.extended_barrel_weapon != None)
    has_eb = weapons.filter(has_or_is_eb, lambda w: w.extended_barrel_weapon != None)
    is_eb = weapons.filter(has_or_is_eb, lambda w: w.is_extended_barrel)

    pd.options.mode.chained_assignment, old = None, pd.options.mode.chained_assignment
    has_or_is_eb.loc[has_eb.index] -= has_eb.values
    has_or_is_eb.loc[is_eb.index] -= has_eb.values
    pd.options.mode.chained_assignment = old

    target = (is_eb - has_eb.values).abs()
    source = has_or_is_eb


styler = (selected_illustration(weapons, target, source)
          .format({col: lambda x: f"{x:.1f}".rstrip('0').rstrip('.') for col in target.select_dtypes(include='float').columns})
          #.format_index({name : w.display_name for name, w in weapons.weapons.items()}, axis=0)
          .format_index(lambda x: f"row_{x}", axis=0)
          )

#https://discuss.streamlit.io/t/select-all-on-a-streamlit-multiselect/9799

# maybe use aggrid for styling index labels

with st.container():
    st.table(styler)