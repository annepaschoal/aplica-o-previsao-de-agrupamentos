import streamlit as st
import pandas as pd
import joblib
import os

# --- CARREGAMENTO DOS MODELOS (Com caminho seguro para WSL) ---
caminho = os.path.dirname(__file__)

encoder = joblib.load(os.path.join(caminho, 'encoder.pkl'))
scaler = joblib.load(os.path.join(caminho, 'scaler.pkl'))
kmeans = joblib.load(os.path.join(caminho, 'kmeans.pkl'))

# --- CONFIGURAÇÃO DA PÁGINA ---
st.title('Grupos de interesse para marketing')

st.write("""
         Neste projeto, aplicamos o algoritmo de clusterização K-means para identificar e prever agrupamentos de interesses de usuários, com o objetivo de direcionar campanhas de marketing de forma mais eficaz.
         Através dessa análise, conseguimos segmentar o público em bolhas de interesse, permitindo a criação de campanhas personalizadas e mais assertivas, com base nos padrões de comportamento e preferências de cada grupo.
         """)

# --- FUNÇÃO DE PROCESSAMENTO ---
def processar_prever(df):
    encoded_sexo = encoder.transform(df[['sexo']])
    encoded_df = pd.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(['sexo']))
    dados = pd.concat([df.drop('sexo', axis=1), encoded_df], axis=1)
        
    dados_escalados = scaler.transform(dados)
    cluster = kmeans.predict(dados_escalados)

    return cluster

# --- BOTÃO DE UPLOAD ---
up_file = st.file_uploader('Escolha um arquivo CSV para realizar a previsão', type='csv')

# --- LÓGICA APÓS O UPLOAD ---
if up_file is not None:
    # 1. Descrição dos grupos para o usuário
    st.write("""
                    ### Descrição dos Grupos:
                    - **Grupo 0** é focado em um público jovem com forte interesse em moda, música e aparência.
                    - **Grupo 1** está muito associado a esportes, especialmente futebol americano, basquete e atividades culturais como banda e rock.
                    - **Grupo 2** é mais equilibrado, com interesses em música, dança, e moda.
                """)
    
    # 2. Transforma o CSV em Dataframe e faz a previsão
    df = pd.read_csv(up_file)
    cluster = processar_prever(df)
    
    # 3. Insere a resposta do modelo na primeira coluna
    df.insert(0, 'grupos', cluster)
    
    # 4. Mostra o resultado na tela
    st.write('Visualização dos resultados (10 primeiros registros):')
    st.write(df.head(10))
    
    # 5. Prepara e oferece o botão de download
    csv = df.to_csv(index=False)
    st.download_button(
        label='Baixar resultados completos', 
        data=csv, 
        file_name='Grupos_interesse.csv', 
        mime='text/csv'
    )