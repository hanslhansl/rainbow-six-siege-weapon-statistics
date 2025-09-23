import create_weapon_statistics_file as cwsf, streamlit as st, pandas as pd, altair as alt, sys

@st.cache_resource
def get_weapons():
    return cwsf.Weapons()

# Load data
ws = get_weapons()

# Streamlit UI
st.set_page_config(
    page_title="r6s weapon stats",
    page_icon=":material/looks_6:", # counter_6 looks_6
    layout="wide",
)
st.markdown(f"## [rainbow six siege weapon statistics]({cwsf.github_link})")
st.markdown("interactive visualisation of damage data")

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

    additional_parameter_indices = st.pills(
        label=f"choose {selected_stat.additional_parameter_name} level",
        options=range(len(selected_stat.additional_parameters)),
        selection_mode="multi",
        #default=0,
        format_func=lambda x: selected_stat.additional_parameters[x][1],
        disabled=len(selected_stat.additional_parameters)==1,
        label_visibility="collapsed",
        width="stretch"
        )
    if len(additional_parameter_indices):
        additional_parameter = [selected_stat.additional_parameters[i] for i in additional_parameter_indices]
    else:
        additional_parameter = selected_stat.additional_parameters

"""
### choose weapons
"""
with st.container(border=True):
    css_lines = [
        "<style>",
        # optional base override to make colors more visible
        "span[data-baseweb='tag']{ color: black !important; }",
    ]
    for opt in ws.base_weapons:
        # escape any double quotes in the label
        safe = opt.replace('"', '\\"')
        css_lines.append(
            f'span[data-baseweb="tag"]:has(span[title="{safe}"]) {{'
            f'  background-color: {ws.base_weapons[opt].color.to_css()} !important;'
            f'  color: black !important;'
            f'}}'
        )
    css_lines.append("</style>")
    st.markdown("\n".join(css_lines), unsafe_allow_html=True)

    # weapon multiselect
    selected_weapons = st.multiselect(
        label="select weapon(s)",
        options=ws.base_weapons,
        key="selected_weapons_multiselect"
        )

    # checkbox whether to show eb stats
    include_eb = st.checkbox(
        label="include extended barrel stats",
        value=True
        )
    
    # Create rows of buttons for weapon classes
    def add_class_to_selection(class_):
        st.session_state.selected_weapons_multiselect = list(dict.fromkeys(
            st.session_state.selected_weapons_multiselect + [name for name, w in ws.base_weapons.items() if w.class_==class_]
            ))
    for inner_col, class_ in zip(st.columns(len(cwsf.Weapon.classes)), cwsf.Weapon.classes):
        with inner_col:
            st.button(class_, on_click=lambda c=class_: add_class_to_selection(c))

    if len(selected_weapons):
        if include_eb:
            for i in range(len(selected_weapons) - 1, -1, -1):
                w = ws.base_weapons[selected_weapons[i]]
                if w.extended_barrel_weapon:
                    selected_weapons.insert(i + 1, w.extended_barrel_weapon.name)
        selected_weapons = tuple(selected_weapons)
    elif include_eb:
        selected_weapons = None # all weapons
    else:
        selected_weapons = tuple(ws.base_weapons)

    
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

if extended_barrel_difference:
    data = ws.vectorize_and_interleave(ws.extended_barrel_difference(selected_stat.stat_method), additional_parameter, selected_weapons)
else:
    data = ws.vectorize_and_interleave(selected_stat.stat_method, additional_parameter, selected_weapons)

styler = selected_illustration(ws, data, additional_parameter)

# maybe use aggrid for styling index labels


st.markdown("""
    <style>
        table td {
            padding: 1px !important;
            text-align: center !important;
        }
        table th {
            padding: 1px !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.table(styler)