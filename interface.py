import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import logging
from pathlib import Path
from PIL import Image
import os
import pickle

# Configuración de la página
st.set_page_config(
    page_title="TuPapiCuyFilms - Sistema de Recomendación",
    page_icon="🎬",
    layout="wide"
)

# Configuración de logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

@st.cache_resource
def load_pretrained_models():
    """Cargar modelos pre-entrenados desde archivos .pkl"""
    try:
        # Cargar K-Means
        with open('trained_models/centroids_KMEANS.pkl', 'rb') as f:
            kmeans_data = pickle.load(f)
        
        # Cargar GMM
        with open('trained_models/centroids_GMM.pkl', 'rb') as f:
            gmm_data = pickle.load(f)
        
        print("✓ Modelos pre-entrenados cargados correctamente")
        return kmeans_data, gmm_data
        
    except Exception as e:
        st.error(f"❌ Error cargando modelos pre-entrenados: {str(e)}")
        return None, None

def predict_kmeans_cluster(features, centroids):
    """Predecir cluster usando centroides de K-Means"""
    features = np.array(features, dtype=np.float64).reshape(1, -1)
    distances = cdist(features, centroids, metric='euclidean')
    return np.argmin(distances, axis=1)[0]

def predict_gmm_cluster(features, means, covariances, weights):
    """Predecir cluster usando parámetros de GMM"""
    features = np.array(features, dtype=np.float64).reshape(1, -1)
    
    # Calcular responsabilidades
    responsibilities = np.zeros(len(means))
    for k in range(len(means)):
        # Distribución normal multivariada
        n_features = features.shape[1]
        det = np.linalg.det(covariances[k])
        if det <= 0:
            det = 1e-6
        norm = 1.0 / (np.power(2 * np.pi, n_features / 2) * np.sqrt(det))
        inv_cov = np.linalg.inv(covariances[k])
        diff = features - means[k]
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        responsibilities[k] = weights[k] * norm * np.exp(exponent)
    
    # Normalizar y retornar cluster con mayor probabilidad
    responsibilities /= (responsibilities.sum() + 1e-10)
    return np.argmax(responsibilities)

@st.cache_resource
def load_data():
    """Cargar datos de test y modelos pre-entrenados"""
    try:
        # Cargar características HOG_lda_18d de test
        features_path = 'features_test/HOG_lda_18d.csv'
        if not os.path.exists(features_path):
            st.error(f"❌ Archivo no encontrado: {features_path}")
            return None, None, None
        
        features_df = pd.read_csv(features_path)
        
        # Convertir características a float64
        feature_columns = [col for col in features_df.columns if col.startswith('comp_')]
        for col in feature_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        features_df = features_df.dropna()
        
        print(f"✓ Características cargadas: {features_df.shape}")
        
        # Cargar películas de test
        test_movies_path = 'movies_test.csv'
        if not os.path.exists(test_movies_path):
            st.error(f"❌ Archivo no encontrado: {test_movies_path}")
            return None, None, None
            
        test_movies = pd.read_csv(test_movies_path)
        
        # Combinar datos
        movies_data = pd.merge(test_movies, features_df, on='movieId', how='inner')
        
        if len(movies_data) == 0:
            st.error("❌ No hay datos coincidentes")
            return None, None, None
        
        print(f"✓ Dataset combinado: {len(movies_data)} películas")
        
        # Cargar modelos pre-entrenados
        kmeans_data, gmm_data = load_pretrained_models()
        
        if kmeans_data is None or gmm_data is None:
            return None, None, None
        
        # Predecir clusters para todas las películas de test usando modelos pre-entrenados
        X = movies_data[feature_columns].values.astype(np.float64)
        
        # Predecir con K-Means
        kmeans_labels = []
        for i in range(len(X)):
            cluster = predict_kmeans_cluster(X[i], kmeans_data['centroids'])
            kmeans_labels.append(cluster)
        movies_data['kmeans_cluster'] = kmeans_labels
        
        # Predecir con GMM
        gmm_labels = []
        for i in range(len(X)):
            cluster = predict_gmm_cluster(X[i], gmm_data['means'], 
                                         gmm_data['covariances'], gmm_data['weights'])
            gmm_labels.append(cluster)
        movies_data['gmm_cluster'] = gmm_labels
        
        print(f"✓ Clusters asignados correctamente")
        
        return movies_data, kmeans_data, gmm_data
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        print(f"ERROR: {traceback.format_exc()}")
        return None, None, None

def get_poster_path(movie_id):
    """Obtener ruta del poster"""
    poster_path = f"posters_test/{movie_id}.jpg"
    return poster_path if os.path.exists(poster_path) else None

def display_poster(movie_id, title, width=200):
    """Mostrar poster de película"""
    poster_path = get_poster_path(movie_id)
    if poster_path:
        try:
            image = Image.open(poster_path)
            st.image(image, caption=title, width=width)
        except:
            st.error(f"❌ Error cargando poster")
    else:
        st.warning(f"📭 Poster no encontrado")

def log_search(movie_title):
    """Registrar búsqueda en log"""
    logging.info(f"BUSQUEDA: {movie_title}")

def get_search_stats():
    """Obtener estadísticas de búsquedas"""
    try:
        with open('logs/app.log', 'r') as f:
            lines = f.readlines()
        
        search_counts = {}
        for line in lines:
            if 'BUSQUEDA:' in line:
                movie = line.split('BUSQUEDA: ')[1].strip()
                search_counts[movie] = search_counts.get(movie, 0) + 1
        return search_counts
    except:
        return {}

def main():
    # INICIALIZACIÓN ROBUSTA
    if 'search_result' not in st.session_state:
        st.session_state.search_result = None
    if 'model_state' not in st.session_state:
        st.session_state.model_state = True
    if 'search_movie_id' not in st.session_state:
        st.session_state.search_movie_id = None
    
    # Cargar datos y modelos pre-entrenados
    movies_data, kmeans_data, gmm_data = load_data()
    
    if movies_data is None:
        st.error("❌ Error crítico: No se pudieron cargar los datos")
        return
    
    # Sidebar
    with st.sidebar:
        if os.path.exists('TuPapiCuyFilms.png'):
            st.image('TuPapiCuyFilms.png', width=200)
        else:
            st.title("🎬 TuPapiCuyFilms")
        
        st.markdown("---")
        st.subheader("🔧 Configuración del Modelo")
        
        if st.button("🔁 Cambiar Modelo", use_container_width=True):
            st.session_state.model_state = not st.session_state.model_state
            st.rerun()
        
        current_model = "K-means" if st.session_state.model_state else "GMM"
        st.success(f"**Modelo actual:** {current_model}")
        
        st.markdown("---")
        st.subheader("📊 Estadísticas")
        search_stats = get_search_stats()
        for movie, count in list(search_stats.items())[:5]:
            st.write(f"• {movie}: {count}")
    
    # Contenido principal
    st.title("🎭 Sistema de Recomendación de Películas")
    st.markdown("---")
    
    # Búsqueda
    col1, col2 = st.columns([2, 1])
    with col1:
        search_option = st.radio("Buscar por:", ["Título", "Movie ID"], horizontal=True)
    
    with col2:
        if st.button("🔄 Volver a Vista General", use_container_width=True):
            st.session_state.search_result = None
            st.session_state.search_movie_id = None
            st.rerun()
    
    # Selector de película
    if search_option == "Título":
        selected_title = st.selectbox("Selecciona una película:", movies_data['title'].tolist())
        search_movie_df = movies_data[movies_data['title'] == selected_title]
    else:
        selected_id = st.selectbox("Selecciona un Movie ID:", movies_data['movieId'].tolist())
        search_movie_df = movies_data[movies_data['movieId'] == selected_id]
    
    if not search_movie_df.empty:
        search_movie = search_movie_df.iloc[0]
        
        if st.button("🔍 Buscar Recomendaciones", type="primary", use_container_width=True):
            st.session_state.search_movie_id = search_movie['movieId']
            log_search(search_movie['title'])
            st.rerun()
    
    # Mostrar recomendaciones
    if st.session_state.search_movie_id is not None:
        current_movie_df = movies_data[movies_data['movieId'] == st.session_state.search_movie_id]
        if not current_movie_df.empty:
            show_recommendations(current_movie_df.iloc[0], movies_data, kmeans_data, gmm_data)
        else:
            st.error("❌ Película no encontrada")
            st.session_state.search_movie_id = None
            st.rerun()
    else:
        show_default_view(movies_data)

def show_default_view(movies_data):
    """Vista por defecto"""
    st.subheader("🎥 Cartelera de Películas")
    st.write("Selecciona una película para obtener recomendaciones")
    
    random_movies = movies_data.sample(min(10, len(movies_data)))
    cols = st.columns(5)
    for idx, (_, movie) in enumerate(random_movies.iterrows()):
        with cols[idx % 5]:
            display_poster(movie['movieId'], movie['title'], 150)
            st.caption(movie['title'])

def show_recommendations(search_movie, movies_data, kmeans_data, gmm_data):
    """Mostrar recomendaciones usando modelos pre-entrenados"""
    if search_movie is None or not isinstance(search_movie, pd.Series):
        st.error("❌ Error: Datos inválidos")
        return
    
    try:
        title = search_movie['title']
        movie_id = search_movie['movieId']
        st.subheader(f"🎯 Recomendaciones para: {title}")
    except:
        st.error("❌ Error: No se puede acceder a los datos de la película")
        return
    
    try:
        # Obtener características de la película buscada
        feature_columns = [col for col in movies_data.columns if col.startswith('comp_')]
        search_features = search_movie[feature_columns].values.astype(np.float64)
        
        # Determinar cluster según modelo seleccionado
        if st.session_state.model_state:
            # K-Means
            current_cluster = predict_kmeans_cluster(search_features, kmeans_data['centroids'])
            cluster_movies = movies_data[movies_data['kmeans_cluster'] == current_cluster]
            model_name = "K-means"
        else:
            # GMM
            current_cluster = predict_gmm_cluster(search_features, gmm_data['means'], 
                                                 gmm_data['covariances'], gmm_data['weights'])
            cluster_movies = movies_data[movies_data['gmm_cluster'] == current_cluster]
            model_name = "GMM"
        
        # Excluir película actual
        cluster_movies = cluster_movies[cluster_movies['movieId'] != movie_id]
        
        # Calcular similitudes (distancia euclidiana inversa)
        similarities = []
        for _, movie in cluster_movies.iterrows():
            movie_features = movie[feature_columns].values.astype(np.float64)
            distance = np.linalg.norm(search_features - movie_features)
            similarity = 1 / (1 + distance)
            similarities.append((movie['movieId'], movie['title'], similarity))
        
        # Ordenar por similitud y tomar top 10
        similarities.sort(key=lambda x: x[2], reverse=True)
        top_recommendations = similarities[:10]
        
        # Mostrar resultados
        st.info(f"**Modelo:** {model_name} | **Cluster:** {current_cluster} | **Encontradas:** {len(top_recommendations)}")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("🎬 Película Buscada")
            display_poster(movie_id, title, 250)
            st.success(f"**{title}**")
            st.write(f"**ID:** {movie_id}")
            st.write(f"**Cluster:** {current_cluster}")
        
        with col2:
            st.subheader(f"📋 Similares (Top {len(top_recommendations)})")
            
            if not top_recommendations:
                st.warning("No hay películas similares en este cluster")
                return
            
            rows = [st.columns(5) for _ in range(2)]
            for idx, (rec_id, rec_title, similarity) in enumerate(top_recommendations[:10]):
                col_idx = idx % 5
                row_idx = idx // 5
                if row_idx < len(rows):
                    with rows[row_idx][col_idx]:
                        st.markdown(f"<div style='text-align: center; background: #ff4b4b; color: white; border-radius: 10px; padding: 2px; font-size: 12px; margin-bottom: 5px;'>#{idx+1}</div>", unsafe_allow_html=True)
                        display_poster(rec_id, rec_title, 150)
                        st.caption(rec_title)
                        st.progress(float(similarity), text=f"Sim: {similarity:.2f}")
                        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()