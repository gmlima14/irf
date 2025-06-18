import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo  # disponível a partir do Python 3.9
from io import BytesIO
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="IRF - Análise de Fornecedores", layout="wide")
st.title("🔍 IRF - Índice de Risco de Fornecedores")

uploaded_pedidos = st.file_uploader("📤 Envie a planilha de pedidos em aberto", type=["xlsx", "csv"])
uploaded_modelo = st.file_uploader("📤 Envie o modelo treinado (pkl)", type=["pkl"])
uploaded_carga = st.file_uploader("📤 Envie a planilha de carga média de fornecedores", type=["xlsx", "csv"])

if uploaded_pedidos and uploaded_modelo and uploaded_carga:
    # Leitura dos arquivos
    df_pedidos_em_aberto = pd.read_csv(uploaded_pedidos) if uploaded_pedidos.name.endswith('.csv') else pd.read_excel(uploaded_pedidos)
    df_carga = pd.read_csv(uploaded_carga) if uploaded_carga.name.endswith('.csv') else pd.read_excel(uploaded_carga)

    st.success("✅ Arquivos carregados com sucesso!")

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

    # Carrega modelo
    modelo = load_model(uploaded_modelo)

    # Previsões
    previsoes = predict_model(modelo, data=df_pedidos_em_aberto)
    previsoes.rename(columns={
        "prediction_label": "Previsão",
        "prediction_score": "Confiabilidade",
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
        'total_pedidos', 'carga_media', 'taxa_carga',  'valor_total', 'taxa_valor', 'confiabilidade_media',  'indice_risco'
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
    st.info("📄 Aguarde o envio de todos os arquivos necessários.")
