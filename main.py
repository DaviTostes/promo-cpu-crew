import streamlit as st
import requests

from agent import create_instagram_crew

API_URL = "http://localhost:8080"

st.title("Instagram Profile Analyzer")
st.subheader("Gere um briefing para Landing Page a partir de um perfil do Instagram")

username = st.text_input("Digite o @ do perfil (sem o @):", placeholder="ex: fulano.oficial")

if st.button("Analisar Perfil", disabled=not username):
    with st.spinner(f"Analisando o perfil @{username}..."):
        crew = create_instagram_crew(username.strip())
        result = crew.kickoff()
        dados = result.model_dump()

    briefing = dados["raw"]
    st.success(f"Briefing do perfil @{username} gerado com sucesso!")
    st.markdown(briefing)

    with st.spinner("Gerando sua Landing Page... isso pode levar alguns minutos."):
        try:
            response = requests.post(
                f"{API_URL}/create",
                json={"company_info": briefing},
                timeout=300,
            )
            response.raise_for_status()
            url = response.json().get("url", "")
        except requests.RequestException as e:
            st.error(f"Erro ao gerar a Landing Page: {e}")
            st.stop()

    st.success("Landing Page criada com sucesso!")
    st.markdown(f"### Acesse sua Landing Page: [{url}]({url})")
