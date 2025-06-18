import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO
import joblib  # Ou pickle, dependendo de como você salvou seu modelo sklearn
import requests # Para fazer requisições HTTP
import os

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="IRF - Análise de Fornecedores", layout="wide")
st.title("🔍 IRF - Índice de Risco de Fornecedores")

# --- URLs dos arquivos no Google Drive ---
# Exemplo: Substitua com os IDs reais dos seus arquivos

MODELO_DRIVE_ID = st.secrets["MODELO_DRIVE_ID"] # USE ESTA LINHA
CARGA_DRIVE_ID = st.secrets["CARGA_DRIVE_ID"]   # USE ESTA LINHA

MODELO_URL = f"https://drive.google.com/uc?export=download&id={MODELO_DRIVE_ID}"
CARGA_URL = f"https://drive.google.com/uc?export=download&id={CARGA_DRIVE_ID}"

@st.cache_data # Use st.cache_resource para o modelo se ele for um objeto grande e não um DataFrame
def load_data_from_drive(url, file_type):
    st.write(f"Baixando arquivo de: {url}") # Para depuração
    try:
        response = requests.get(url)
        response.raise_for_status() # Levanta um erro para códigos de status HTTP ruins (4xx ou 5xx)

        if file_type == 'excel':
            return pd.read_excel(BytesIO(response.content))
        elif file_type == 'csv':
            return pd.read_csv(BytesIO(response.content))
        elif file_type == 'pkl':
            return joblib.load(BytesIO(response.content))
        else:
            st.error("Tipo de arquivo não suportado para download.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar o arquivo do Drive: {e}. Verifique o ID e as permissões de compartilhamento.")
        return None
    except Exception as e:
        st.error(f"Erro ao processar o arquivo baixado: {e}")
        return None

# Carrega os arquivos automaticamente
modelo = load_data_from_drive(MODELO_URL, 'pkl')
df_carga = load_data_from_drive(CARGA_URL, 'excel') # Assumindo que sua carga é Excel

# Remova os uploaders, já que os arquivos serão baixados automaticamente
# uploaded_pedidos = st.file_uploader("📤 Envie a planilha de pedidos em aberto", type=["xlsx", "csv"])
# uploaded_modelo = st.file_uploader("📤 Envie o modelo treinado (pkl)", type=["pkl"])
# uploaded_carga = st.file_uploader("📤 Envie a planilha de carga média de fornecedores", type=["xlsx", "csv"])

# O uploader para pedidos em aberto ainda pode ser útil
uploaded_pedidos = st.file_uploader("📤 Envie a planilha de pedidos em aberto", type=["xlsx", "csv"])

if uploaded_pedidos and modelo is not None and df_carga is not None:
    # Leitura dos arquivos
    df_pedidos_em_aberto = pd.read_csv(uploaded_pedidos) if uploaded_pedidos.name.endswith('.csv') else pd.read_excel(uploaded_pedidos)

    st.success("✅ Arquivos carregados e baixados com sucesso!")

    # ... (o restante do seu código permanece o mesmo) ...

    # Conversão de tipos
    df_pedidos_em_aberto['MATKL'] = df_pedidos_em_aberto['MATKL'].astype('category')
    df_pedidos_em_aberto['Vendor'] = df_pedidos_em_aberto['Vendor'].astype('category')
    df_pedidos_em_aberto["BEDAT"] = pd.to_datetime(df_pedidos_em_aberto["BEDAT"], errors="coerce")
    df_pedidos_em_aberto["Due Date (incl. ex works time)"] = pd.to_datetime(df_pedidos_em_aberto["Due Date (incl. ex works time)"], errors="coerce")

    hoje = datetime.today()
    df_pedidos_em_aberto["MesPedido"] = df_pedidos_em_aberto["BEDAT"].dt.month
    df_pedidos_em_aberto["IdadePedido"] = (hoje - df_pedidos_em_aberto["BEDAT"]).dt.days
    df_pedidos_em_aberto["DiasParaEntrega"] = (df_pedidos_em_aberto["Due Date (incl. ex works time)"] - df_pedidos_em_aberto["BEDAT"]).dt.days

    # Carga
    pedidos_abertos_por_fornecedor = df_pedidos_em_aberto['Vendor'].value_counts()
    df_pedidos_em_aberto['carga_fornecedor'] = df_pedidos_em_aberto['Vendor'].map(pedidos_abertos_por_fornecedor).fillna(0).astype(int)

    # Preparar os dados para o modelo sklearn
    features_do_modelo = [
        "MesPedido",
        "IdadePedido",
        "DiasParaEntrega",
        "carga_fornecedor",
        "NetOrderValue"
        # Adicione outras features numéricas ou codificadas que seu modelo usa
    ]

    # Certifique-se de que todas as colunas necessárias existam no DataFrame antes de selecioná-las
    # E que o pré-processamento (e.g., OneHotEncoder para categorias) seja aplicado aqui,
    # se não estiver embutido no seu pipeline do modelo.
    # Exemplo: tratamento de colunas categóricas (ajuste conforme seu modelo)
    # df_pedidos_em_aberto = pd.get_dummies(df_pedidos_em_aberto, columns=['MATKL', 'Vendor'], drop_first=True) # Exemplo

    # Remova colunas que não são features para o modelo ANTES de passar para o predict
    # Certifique-se que X_predict tenha as mesmas colunas e ordem que no treinamento
    # Você pode precisar de um pipeline ou transformador para garantir a ordem e as colunas.
    try:
        X_predict = df_pedidos_em_aberto[features_do_modelo]
    except KeyError as e:
        st.error(f"Coluna faltando no DataFrame para a previsão do modelo: {e}. Verifique as 'features_do_modelo'.")
        st.stop() # Para o script se colunas essenciais estiverem faltando

    # Previsões
    previsoes_labels = modelo.predict(X_predict)
    previsoes_proba = modelo.predict_proba(X_predict)

    df_pedidos_em_aberto['Previsão'] = previsoes_labels
    df_pedidos_em_aberto['Confiabilidade'] = [prob[label] for prob, label in zip(previsoes_proba, previsoes_labels)]

    previsoes = df_pedidos_em_aberto
    previsoes.rename(columns={
        "carga_fornecedor": "Carga do Fornecedor",
        "EBELN": "PO",
        "EBELP": "Item",
        "BEDAT": "Data de Emissão da PO",
        "Due Date (incl. ex works time)": "Stat. Del. Date",
        "Material Text (AST or Short Text)": "Descrição do Item",
        "Vendor Name": "Fornecedor",
        "MATKL": "Material Number",
        "NetOrderValue": "Valor Net",
        "MesPedido": "Mês do Pedido",
        "IdadePedido": "Idade do Pedido",
        "DiasParaEntrega": "Dias para Entrega",
    }, inplace=True)
    previsoes["Previsão"] = previsoes["Previsão"].replace({0: "No Prazo", 1: "Atraso"})

    # Agrupamento (restante do seu código)
    agrupado = previsoes.groupby('Vendor').agg(
        pedidos_no_prazo=('Previsão', lambda x: (x == 'No Prazo').sum()),
        pedidos_atrasados=('Previsão', lambda x: (x == 'Atraso').sum()),
        total_pedidos=('Previsão', 'count'),
        confiabilidade_media=('Confiabilidade', 'mean'),
        valor_total=('Valor Net', 'sum'),
        valor_atrasado=('Valor Net', lambda x: (x[previsoes.loc[x.index, 'Previsão'] == 'Atraso']).sum()),
        valor_no_prazo=('Valor Net', lambda x: (x[previsoes.loc[x.index, 'Previsão'] == 'No Prazo']).sum()),
        fornecedor=('Fornecedor', 'first')
    ).reset_index()

    df_carga.rename(columns={'carga_fornecedor': 'carga_media'}, inplace=True)
    agrupado = agrupado.merge(df_carga[['Vendor', 'carga_media']], on='Vendor', how='left')

    agrupado['taxa_no_prazo'] = agrupado['pedidos_no_prazo'] / agrupado['total_pedidos']
    agrupado['taxa_valor'] = agrupado['valor_no_prazo'] / agrupado['valor_total']

    def calcular_taxa_carga(row):
        carga_media = row['carga_media']
        total_pedidos = row['total_pedidos']
        if pd.isna(carga_media) or carga_media <= 2:
            return 1
        taxa = total_pedidos / carga_media
        return min(max(taxa, 1), 1.5)

    agrupado['taxa_carga'] = agrupado.apply(calcular_taxa_carga, axis=1)
    agrupado['indice_bruto'] = agrupado['taxa_no_prazo'] * agrupado['confiabilidade_media'] * agrupado['taxa_valor'] / agrupado['taxa_carga']
    agrupado['indice_risco'] = (1 - agrupado['indice_bruto'])*100
    agrupado = agrupado.round(2)
    agrupado['carga_media'] = agrupado['carga_media'].apply(lambda x: int(x) if x >= 1 else 0)

    df_fornecedores = agrupado[[
        'fornecedor', 'Vendor', 'pedidos_no_prazo', 'pedidos_atrasados', 'taxa_no_prazo',
        'total_pedidos', 'carga_media', 'taxa_carga',  'valor_total', 'taxa_valor', 'confiabilidade_media',  'indice_risco'
    ]]

    df_fornecedores = df_fornecedores.sort_values('indice_risco', ascending=False).reset_index(drop=True)
    df_fornecedores['Ranking'] = df_fornecedores.index + 1
    df_fornecedores = df_fornecedores[['Ranking'] + [col for col in df_fornecedores.columns if col != 'Ranking']]

    df_fornecedores = df_fornecedores.rename(columns={
        'fornecedor': 'Fornecedor',
        'pedidos_no_prazo': 'PO previstas no prazo',
        'pedidos_atrasados': 'PO previstas atrasadas',
        'taxa_no_prazo': 'Taxa de PO previstas no prazo',
        'total_pedidos': 'Total de PO',
        'carga_media': 'Carga Média de PO',
        'taxa_carga': 'Taxa de Carga',
        'valor_total': 'Valor NET de PO',
        'taxa_valor': 'Taxa de Valor previsto no prazo',
        'indice_risco': 'Índice de Risco',
        'confiabilidade_media': 'Confiabilidade Média'
    })

    agora = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime('%d-%m-%Y %H-%M')
    caminho_arquivo = f'IRF - {agora}.xlsx'

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_fornecedores.to_excel(writer, sheet_name='Fornecedores', index=False)
        previsoes.to_excel(writer, sheet_name='Pedidos em Aberto', index=False)
    output.seek(0)

    st.download_button(
        label="📥 Baixar arquivo de resultado",
        data=output,
        file_name=caminho_arquivo,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    # Ajuste a mensagem para refletir que os arquivos são baixados automaticamente
    if modelo is None or df_carga is None:
        st.info("⏳ Tentando baixar o modelo e a planilha de carga do Google Drive...")
    if uploaded_pedidos is None:
        st.info("📄 Aguardando o envio da planilha de pedidos em aberto.")