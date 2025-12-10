import streamlit as st

from agent import gpu_deals_crew

st.title("GPU Deals Crew")
st.subheader("Find the best GPU deals available online!")

if st.button("Find Deals"):
    with st.spinner("Searching for the best GPU deals..."):
        result = gpu_deals_crew.kickoff()
        deals = result.model_dump()
    st.success("Here are the best GPU deals we found:")
    st.write(deals["raw"])
