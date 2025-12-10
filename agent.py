from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Task, Crew, Process

llm = "gemini-2.5-flash"

search_agent = Agent(
    role="Buscador de E-commerce",
    goal="Buscar placas de vídeo em e-commerces brasileiros e coletar preços, estoque e links",
    backstory="Expert em busca de produtos online nos principais e-commerces brasileiros. Foca em coletar dados completos rapidamente.",
    verbose=False,
    allow_delegation=False,
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=llm
)

analyst_agent = Agent(
    role="Analista de Preços",
    goal="Analisar ofertas, validar promoções reais e avaliar custo-benefício",
    backstory="Especialista em hardware e análise de preços de GPUs. Identifica promoções falsas e determina o valor real de ofertas.",
    verbose=False,
    allow_delegation=False,
    tools=[],
    llm=llm
)

reporter_agent = Agent(
    role="Gerador de Relatórios",
    goal="Criar relatórios CURTOS e diretos ao ponto, focando apenas nas melhores ofertas",
    backstory="Especialista em síntese. Elimina informações desnecessárias e vai direto ao que importa.",
    verbose=False,
    allow_delegation=False,
    llm=llm
)

search_task = Task(
    description=(
        "Buscar placas de vídeo em Amazon BR e Mercado Livre.\n\n"
        "Coletar: modelo, preço, estoque, link, frete, garantia, reputação do vendedor, cupons.\n"
        "Organizar por site."
    ),
    expected_output="Lista de GPUs com: modelo, loja, preço, cupons, estoque, link, frete, garantia, reputação.",
    agent=search_agent
)

analysis_task = Task(
    description=(
        "Analisar ofertas coletadas: validar promoções reais, calcular desconto real, avaliar custo-benefício.\n\n"
        "Classificar por faixa de preço (Entrada <R$1.500, Intermediária R$1.500-3.500, High-end >R$3.500).\n"
        "Indicar melhor GPU para 1080p, 1440p e 4K.\n"
        "Veredito: COMPRAR/ÓTIMO/JUSTO/ESPERAR/EVITAR.\n"
        "Destacar TOP 3 ofertas."
    ),
    expected_output="Ranking TOP 3 com justificativas. Análise por faixa de preço com vereditos. Recomendações por resolução. Alertas de promoções falsas.",
    agent=analyst_agent,
    context=[search_task]
)

report_task = Task(
    description=(
        "Criar relatório CURTO e fácil de ler:\n"
        "1. Título simples\n"
        "2. TOP 3 melhores ofertas (modelo, preço, link, veredito em 5 linhas no maximo)\n"
        "3. Recomendação rápida por uso (1080p/1440p/4K) - APENAS o nome do modelo\n"
        "4. 1-2 alertas importantes (se houver, se nao, nao coloque este topico)\n\n"
        "5. Consideracoes finais sobre as ofertas.\n\n"
        "6. Links para cada oferta no final do relatório."
    ),
    expected_output="Relatório markdown CURTO: TOP 3 com preço/link/veredito, recomendações rápidas por resolução, alertas breves.",
    agent=reporter_agent,
    context=[analysis_task]
)


gpu_deals_crew = Crew(
    agents=[search_agent, analyst_agent, reporter_agent],
    tasks=[search_task, analysis_task, report_task],
    process=Process.sequential,
    verbose=True,
    memory=False,
    cache=False,
    max_rpm=30
)
