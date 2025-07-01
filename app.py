import streamlit as st
from bs4 import BeautifulSoup
from googlesearch import search
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

# Configuración de Hugging Face
hf_api_key = os.getenv("HF_API_KEY")
hf_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3 " # Cambia esto al modelo que quieras usar
if not hf_api_key:
    st.error("⚠️ No se encontró la clave de Hugging Face. Configurala en .env.")
    st.stop()

# --- Función para buscar en Google ---
def buscar_desde_google(consulta):
    query = f"{consulta} site:mercadolibre.com.uy"
    try:
        urls = list(search(query, num_results=10))
        if not urls:
            st.error("No se obtuvieron resultados de búsqueda.")
            return []
    except Exception as e:
        st.error("Error al buscar en Google.")
        return []

    productos = []
    for url in urls:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')

            nombre_elem = soup.find('h1', class_='ui-pdp-title')
            precio_elem = soup.find('span', class_='andes-money-amount__fraction')

            if nombre_elem and precio_elem:
                nombre = nombre_elem.text.strip()
                precio_texto = precio_elem.text.strip().replace('.', '').replace(',', '')
                try:
                    precio = int(precio_texto)
                except ValueError:
                    precio = 0

                productos.append({
                    'nombre': nombre,
                    'precio': precio
                })
        except Exception as e:
            continue

    return productos

# --- Función para analizar con IA usando Hugging Face ---
def analizar_con_ia(df, consulta):
    if df.empty:
        return "No hay datos para analizar."

    texto = "\n".join([f"{row['nombre']} - ${row['precio']}" for _, row in df.head(5).iterrows()])
    prompt = f"""
Eres un experto en análisis de mercado. 


Productos:
{texto}

Análisis:
"""

    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.8
        }
    }

    try:
        response = requests.post(hf_api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"⚠️ Error con la IA: {response.status_code} - {response.text}"
    except Exception as e:
        return f"⚠️ Error con la IA: {e}"

# --- UI de Streamlit ---
st.set_page_config(page_title="MercadoBotIA 💸", layout="centered")
st.title("🤖 MercadoBotIA - Búsqueda + Análisis con IA")

st.info("""
Ingresa un producto que quieras buscar en MercadoLibre Uruguay.  
El bot hará una búsqueda automática, extraerá precios y analizará con inteligencia artificial.
""")

consulta = st.text_input("¿Qué querés buscar en MercadoLibre?")

if consulta:
    with st.spinner("🔎 Buscando productos..."):
        productos = buscar_desde_google(consulta)

    if productos:
        df = pd.DataFrame(productos)
        df = df[df['precio'] > 0]  # Filtramos los sin precio válido

        if not df.empty:
            st.subheader("📦 Productos encontrados:")
            st.dataframe(df)

            promedio = int(df['precio'].mean())
            st.markdown(f"💰 **Precio promedio:** ${promedio}")

            mas_barato = df.nsmallest(1, 'precio').iloc[0]
            mas_caro = df.nlargest(1, 'precio').iloc[0]
            st.markdown(f"📉 **Producto más barato:** {mas_barato['nombre']} - ${mas_barato['precio']}")
            st.markdown(f"📈 **Producto más caro:** {mas_caro['nombre']} - ${mas_caro['precio']}")

            # Gráfico de precios
            st.subheader("📊 Distribución de precios")
            plt.figure(figsize=(8, 4))
            plt.hist(df['precio'], bins=10, color='skyblue', edgecolor='black')
            plt.xlabel('Precio ($)')
            plt.ylabel('Cantidad')
            plt.title('Distribución de precios')
            st.pyplot(plt)

            with st.spinner("🤖 Consultando a la IA..."):
                respuesta = analizar_con_ia(df, consulta)
            st.subheader("🧠 Análisis de la IA:")
            st.write(respuesta)

            # Botón de descarga
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📁 Descargar CSV",
                data=csv,
                file_name=f"{consulta}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No se encontraron productos con precios válidos.")

    else:
        st.warning("No se encontraron productos para esa búsqueda.")

st.markdown("---")
st.markdown("💬 Hecho con 💙 usando Python, Streamlit, Hugging Face y BeautifulSoup.")