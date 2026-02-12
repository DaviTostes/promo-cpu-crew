import streamlit as st

from agent import create_instagram_crew

st.title("Instagram Profile Analyzer")
st.subheader("Gere um briefing para Landing Page a partir de um perfil do Instagram")

username = st.text_input("Digite o @ do perfil (sem o @):", placeholder="ex: fulano.oficial")

if st.button("Analisar Perfil", disabled=not username):
    with st.spinner(f"Analisando o perfil @{username}..."):
        crew = create_instagram_crew(username.strip())
        result = crew.kickoff()
        dados = result.model_dump()
    st.success(f"Briefing do perfil @{username} gerado com sucesso!")
    st.markdown(dados["raw"])
