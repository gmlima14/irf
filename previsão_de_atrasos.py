import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO
# import joblib  # REMOVA ISSO
import requests
import os

# Importe load_model e predict_model do PyCaret, como no seu código original
from pycaret.classification import load_model, predict_model 


from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="IRF - Análise de Fornecedores", layout="wide")
st.title("🔍 IRF - Índice de Risco de Fornecedores")

MODELO_DRIVE_ID = st.secrets["MODELO_DRIVE_ID"] # USE ESTA LINHA
CARGA_DRIVE_ID = st.secrets["CARGA_DRIVE_ID"]   # USE ESTA LINHA

MODELO_URL = f"https://drive.google.com/uc?export=download&id={MODELO_DRIVE_ID}"
CARGA_URL = f"https://drive.google.com/uc?export=download&id={CARGA_DRIVE_ID}"

# Função para carregar o modelo PyCaret do Drive
@st.cache_resource # Usar cache_resource para o modelo PyCaret
def load_pycaret_model_from_drive(url):
    st.write(f"Baixando modelo PyCaret de: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()

        # A função load_model do PyCaret pode carregar diretamente de um BytesIO
        # ou de um caminho de arquivo. Precisamos salvar temporariamente
        # ou passar o BytesIO de forma que ela aceite.
        # A forma mais robusta é salvar o conteúdo em um arquivo temporário.
        temp_model_path = "temp_pycaret_model.pkl"
        with open(temp_model_path, "wb") as f:
            f.write(response.content)
        
        # Carrega o modelo PyCaret usando a função load_model do PyCaret
        model = load_model(temp_model_path, verbose=False) # verbose=False para reduzir logs
        
        # Opcional: remover o arquivo temporário
        os.remove(temp_model_path)
        
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar o modelo PyCaret do Drive: {e}. Verifique o ID e as permissões.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo PyCaret: {e}. Certifique-se que o modelo foi salvo corretamente com PyCaret e a versão do PyCaret é compatível.")
        return None

# Função para carregar DataFrames (sem alterações significativas)
@st.cache_data
def load_data_from_drive(url, file_type):
    st.write(f"Baixando planilha de: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()

        if file_type == 'excel':
            return pd.read_excel(BytesIO(response.content))
        elif file_type == 'csv':
            return pd.read_csv(BytesIO(response.content))
        else:
            st.error("Tipo de arquivo de dados não suportado para download.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar a planilha do Drive: {e}. Verifique o ID e as permissões de compartilhamento.")
        return None
    except Exception as e:
        st.error(f"Erro ao processar a planilha baixada: {e}")
        return None

# Carrega os arquivos automaticamente
modelo = load_pycaret_model_from_drive(MODELO_URL) # Use a nova função
df_carga = load_data_from_drive(CARGA_URL, 'excel')

uploaded_pedidos = st.file_uploader("📤 Envie a planilha de pedidos em aberto", type=["xlsx", "csv"])

if uploaded_pedidos and modelo is not None and df_carga is not None:
    df_pedidos_em_aberto = pd.read_csv(uploaded_pedidos) if uploaded_pedidos.name.endswith('.csv') else pd.read_excel(uploaded_pedidos)

    st.success("✅ Arquivos carregados e baixados com sucesso!")

    # AQUI ESTÁ A MAIOR MUDANÇA: USANDO predict_model DO PYCARET
    # Você não precisa mais do bloco 'features_do_modelo'
    # O PyCaret gerenciará o pré-processamento automaticamente
    # apenas certifique-se de que 'df_pedidos_em_aberto' tem as colunas originais
    # nas quais o modelo PyCaret foi treinado.

    # df_pedidos_em_aberto['MATKL'] = df_pedidos_em_aberto['MATKL'].astype('category') # PyCaret fará isso se foi no setup
    # df_pedidos_em_aberto['Vendor'] = df_pedidos_em_aberto['Vendor'].astype('category') # PyCaret fará isso se foi no setup
    # Remova conversões de tipo se o PyCaret as gerencia no pipeline
    df_pedidos_em_aberto["BEDAT"] = pd.to_datetime(df_pedidos_em_aberto["BEDAT"], errors="coerce")
    df_pedidos_em_aberto["Due Date (incl. ex works time)"] = pd.to_datetime(df_pedidos_em_aberto["Due Date (incl. ex works time)"], errors="coerce")

    hoje = datetime.today()
    df_pedidos_em_aberto["MesPedido"] = df_pedidos_em_aberto["BEDAT"].dt.month
    df_pedidos_em_aberto["IdadePedido"] = (hoje - df_pedidos_em_aberto["BEDAT"]).dt.days
    df_pedidos_em_aberto["DiasParaEntrega"] = (df_pedidos_em_aberto["Due Date (incl. ex works time)"] - df_pedidos_em_aberto["BEDAT"]).dt.days

    pedidos_abertos_por_fornecedor = df_pedidos_em_aberto['Vendor'].value_counts()
    df_pedidos_em_aberto['carga_fornecedor'] = df_pedidos_em_aberto['Vendor'].map(pedidos_abertos_por_fornecedor).fillna(0).astype(int)

    # Previsões com predict_model do PyCaret
    # O predict_model lida com o pré-processamento.
    # Certifique-se de que o df_pedidos_em_aberto tem o mesmo formato
    # (colunas e tipos) que os dados usados no setup() original do PyCaret.
    previsoes = predict_model(modelo, data=df_pedidos_em_aberto)

    previsoes.rename(columns={
        "prediction_label": "Previsão", # Nomes de coluna padrão do PyCaret
        "prediction_score": "Confiabilidade", # Nomes de coluna padrão do PyCaret
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

    # Agrupamento
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
        'total_pedidos', 'carga_media', 'taxa_carga', 'valor_total', 'taxa_valor', 'confiabilidade_media', 'indice_risco'
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
    if modelo is None or df_carga is None:
        st.info("⏳ Tentando baixar o modelo e a planilha de carga do Google Drive...")
    if uploaded_pedidos is None:
        st.info("📄 Aguardando o envio da planilha de pedidos em aberto.")