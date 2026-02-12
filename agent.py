from dotenv import load_dotenv
load_dotenv()

from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Task, Crew, Process

llm = "gemini-2.5-flash-lite"

search_agent = Agent(
    role="Pesquisador de Perfil Instagram",
    goal="Encontrar e coletar todas as informações públicas de um perfil do Instagram: bio, legendas recentes, estilo visual, hashtags e tom de comunicação",
    backstory=(
        "Especialista em OSINT e pesquisa de redes sociais. "
        "Consegue encontrar informações públicas de perfis do Instagram através de buscas na web, "
        "acessando páginas de perfil, agregadores e caches públicos para extrair bio, legendas e dados relevantes."
    ),
    verbose=False,
    allow_delegation=False,
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=llm
)

analyst_agent = Agent(
    role="Analista de Perfil e Persona",
    goal="Analisar as informações coletadas do perfil e identificar padrões de comunicação, público-alvo, proposta de valor e elementos-chave para uma Landing Page",
    backstory=(
        "Especialista em marketing digital e análise de persona. "
        "Identifica tom de voz, palavras-chave recorrentes, proposta de valor, "
        "dores e desejos do público a partir de conteúdos de redes sociais."
    ),
    verbose=False,
    allow_delegation=False,
    tools=[],
    llm=llm
)

reporter_agent = Agent(
    role="Gerador de Briefing para Landing Page",
    goal="Compilar todas as informações analisadas em um briefing estruturado e pronto para ser usado na criação de uma Landing Page",
    backstory=(
        "Copywriter e estrategista digital. "
        "Transforma dados brutos de perfis em briefings completos e acionáveis "
        "para criação de Landing Pages de alta conversão."
    ),
    verbose=False,
    allow_delegation=False,
    llm=llm
)


def create_instagram_crew(username: str) -> Crew:
    search_task = Task(
        description=(
            f"Pesquisar o perfil do Instagram @{username}.\n\n"
            f"1. Buscar 'site:instagram.com {username}' e variações para encontrar o perfil.\n"
            f"2. Acessar a página do perfil e coletar:\n"
            "   - Bio completa\n"
            "   - Nome exibido\n"
            "   - Número de seguidores/seguindo (se visível)\n"
            "   - Link na bio\n"
            f"3. Buscar '{username} instagram' para encontrar legendas recentes em caches, "
            "agregadores ou sites que indexam conteúdo do Instagram.\n"
            "4. Coletar o máximo de legendas/posts recentes possível.\n"
            "5. Identificar hashtags recorrentes.\n"
        ),
        expected_output=(
            "Compilado com: bio completa, nome exibido, dados numéricos do perfil, "
            "link na bio, lista de legendas recentes com hashtags, e quaisquer outras informações públicas encontradas."
        ),
        agent=search_agent
    )

    analysis_task = Task(
        description=(
            f"Analisar os dados coletados do perfil @{username} e extrair insights para LP:\n\n"
            "1. Tom de voz (formal, informal, motivacional, técnico, etc.)\n"
            "2. Palavras-chave e expressões recorrentes\n"
            "3. Proposta de valor do perfil (o que ele oferece/promete)\n"
            "4. Público-alvo aparente (idade, interesses, nível socioeconômico)\n"
            "5. Dores e desejos que o perfil aborda\n"
            "6. Diferenciais e posicionamento\n"
            "7. CTAs (chamadas para ação) utilizados\n"
            "8. Elementos de prova social (depoimentos, resultados, números)\n"
        ),
        expected_output=(
            "Análise completa com: tom de voz identificado, palavras-chave, proposta de valor, "
            "perfil do público-alvo, dores/desejos, diferenciais, CTAs usados e provas sociais encontradas."
        ),
        agent=analyst_agent,
        context=[search_task]
    )

    report_task = Task(
        description=(
            f"Criar um briefing completo para Landing Page baseado no perfil @{username}:\n\n"
            "O briefing deve conter:\n"
            "1. **Resumo do Perfil** - Quem é, o que faz, bio\n"
            "2. **Legendas e Conteúdos** - Compilado das legendas encontradas (texto integral)\n"
            "3. **Tom de Voz** - Como se comunica, exemplos\n"
            "4. **Proposta de Valor** - O que oferece e por que é diferente\n"
            "5. **Público-Alvo** - Quem ele atinge\n"
            "6. **Dores e Desejos** - O que o público sente/quer\n"
            "7. **Sugestões para LP** - Headlines, subheadlines, CTAs e seções recomendadas\n"
            "8. **Palavras-chave para Copy** - Lista de termos e expressões para usar na LP\n\n"
            "Escrever tudo em português brasileiro. Ser detalhado e acionável."
        ),
        expected_output=(
            "Briefing em markdown com todas as seções listadas, pronto para ser usado "
            "como base para criação de uma Landing Page."
        ),
        agent=reporter_agent,
        context=[analysis_task]
    )

    return Crew(
        agents=[search_agent, analyst_agent, reporter_agent],
        tasks=[search_task, analysis_task, report_task],
        process=Process.sequential,
        verbose=True,
        memory=False,
        cache=False,
        max_rpm=30
    )
