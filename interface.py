import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from PIL import Image
import os

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

class KMeans:
    """K-Means clustering from scratch"""
    
    def __init__(self, n_clusters=10, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
    
    def fit(self, X):
        """Fit K-Means"""
        # CONVERSIÓN GARANTIZADA a float64
        X = np.array(X, dtype=np.float64, copy=True)
        
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()
        
        for iteration in range(self.max_iter):
            # Assign points to nearest centroid
            distances = cdist(X, self.centroids, metric='euclidean')
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) 
                                     for k in range(self.n_clusters)])
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            
            self.centroids = new_centroids
        
        self.labels_ = labels
        self.inertia_ = np.sum([np.sum((X[labels == k] - self.centroids[k])**2) 
                                for k in range(self.n_clusters)])
        
        return self
    
    def predict(self, X):
        """Predict cluster labels"""
        # CONVERSIÓN GARANTIZADA a float64
        X = np.array(X, dtype=np.float64, copy=True)
        distances = cdist(X, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

class GMM:
    """Gaussian Mixture Model from scratch"""
    
    def __init__(self, n_components=10, max_iter=100, tol=1e-3, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.labels_ = None
    
    def fit(self, X):
        """Fit GMM"""
        # CONVERSIÓN GARANTIZADA a float64
        X = np.array(X, dtype=np.float64, copy=True)
        
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        for iteration in range(self.max_iter):
            # E-step: Calculate responsibilities
            responsibilities = self._calculate_responsibilities(X)
            
            # M-step: Update parameters
            new_weights = responsibilities.mean(axis=0)
            new_means = np.array([
                np.sum(responsibilities[:, k].reshape(-1, 1) * X, axis=0) / responsibilities[:, k].sum()
                for k in range(self.n_components)
            ])
            
            # Check convergence
            if np.abs(new_weights - self.weights_).max() < self.tol:
                break
                
            self.weights_ = new_weights
            self.means_ = new_means
        
        self.labels_ = np.argmax(responsibilities, axis=1)
        return self
    
    def _calculate_responsibilities(self, X):
        """Calculate responsibilities for E-step"""
        responsibilities = np.zeros((len(X), self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self._multivariate_normal(X, self.means_[k], self.covariances_[k])
        return responsibilities / responsibilities.sum(axis=1, keepdims=True)
    
    def _multivariate_normal(self, X, mean, cov):
        """Multivariate normal distribution"""
        n_features = X.shape[1]
        det = np.linalg.det(cov)
        norm = 1.0 / (np.power(2 * np.pi, n_features / 2) * np.sqrt(det))
        inv_cov = np.linalg.inv(cov)
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        return norm * np.exp(exponent)
    
    def predict(self, X):
        """Predict cluster labels"""
        # CONVERSIÓN GARANTIZADA a float64
        X = np.array(X, dtype=np.float64, copy=True)
        responsibilities = self._calculate_responsibilities(X)
        return np.argmax(responsibilities, axis=1)

def load_and_validate_features(file_path):
    """Cargar y validar características garantizando float64"""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Archivo cargado: {df.shape}")
        
        # CONVERSIÓN AGRESIVA a float64
        feature_columns = [col for col in df.columns if col.startswith('comp_')]
        
        for col in feature_columns:
            # Conversión forzada con manejo de errores
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        # VERIFICACIÓN FINAL
        for col in feature_columns:
            if df[col].dtype != np.float64:
                df[col] = df[col].astype(np.float64)
        
        print(f"✓ Tipos de datos finales: {df[feature_columns].dtypes.unique()}")
        return df
        
    except Exception as e:
        print(f"❌ Error cargando {file_path}: {e}")
        return None

@st.cache_resource
def load_data():
    """Cargar datos de test"""
    try:
        # Cargar características con validación
        features_df = load_and_validate_features('features_test/HOG_lda_18d.csv')
        if features_df is None:
            st.error("❌ Error cargando características")
            return None, None, None
        
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
        
        # Preparar características
        feature_columns = [col for col in features_df.columns if col.startswith('comp_')]
        X = movies_data[feature_columns].values
        
        # CONVERSIÓN FINAL GARANTIZADA
        X = np.array(X, dtype=np.float64, copy=True)
        print(f"✓ Tipo final de datos: {X.dtype}")
        
        # Entrenar modelos
        kmeans_model = KMeans(n_clusters=10, random_state=42)
        kmeans_model.fit(X)
        
        gmm_model = GMM(n_components=10, random_state=42)
        gmm_model.fit(X)
        
        # Agregar clusters
        movies_data = movies_data.copy()
        movies_data['kmeans_cluster'] = kmeans_model.labels_
        movies_data['gmm_cluster'] = gmm_model.labels_
        
        print(f"✓ Sistema listo: {len(movies_data)} películas")
        return movies_data, kmeans_model, gmm_model
        
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
    
    # Cargar datos
    movies_data, kmeans_model, gmm_model = load_data()
    
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
    
    # Búsqueda - ENFOQUE ALTERNATIVO: Usar movieId en session_state
    col1, col2 = st.columns([2, 1])
    with col1:
        search_option = st.radio("Buscar por:", ["Título", "Movie ID"], horizontal=True)
    
    with col2:
        if st.button("🔄 Volver a Vista General", use_container_width=True):
            # LIMPIAR COMPLETAMENTE el estado
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
            # GUARDAR SOLO EL ID para evitar problemas con session_state
            st.session_state.search_movie_id = search_movie['movieId']
            log_search(search_movie['title'])
            st.rerun()
    
    # LÓGICA MEJORADA: Recuperar película desde el DataFrame usando ID
    if st.session_state.search_movie_id is not None:
        # Buscar la película actual desde los datos frescos
        current_movie_df = movies_data[movies_data['movieId'] == st.session_state.search_movie_id]
        if not current_movie_df.empty:
            show_recommendations(current_movie_df.iloc[0], movies_data, kmeans_model, gmm_model)
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

def show_recommendations(search_movie, movies_data, kmeans_model, gmm_model):
    """Mostrar recomendaciones - VERSIÓN ROBUSTA"""
    # VERIFICACIÓN EXTREMA
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
        # Obtener características con conversión garantizada
        feature_columns = [col for col in movies_data.columns if col.startswith('comp_')]
        search_features = search_movie[feature_columns].values.reshape(1, -1)
        
        # CONVERSIÓN ABSOLUTAMENTE GARANTIZADA
        search_features = np.array(search_features, dtype=np.float64, copy=True)
        
        # Predecir cluster
        if st.session_state.model_state:
            current_cluster = kmeans_model.predict(search_features)[0]
            cluster_movies = movies_data[movies_data['kmeans_cluster'] == current_cluster]
            model_name = "K-means"
        else:
            current_cluster = gmm_model.predict(search_features)[0]
            cluster_movies = movies_data[movies_data['gmm_cluster'] == current_cluster]
            model_name = "GMM"
        
        # Excluir película actual y calcular similitudes
        cluster_movies = cluster_movies[cluster_movies['movieId'] != movie_id]
        
        similarities = []
        search_features_flat = search_features.flatten()
        
        for _, movie in cluster_movies.iterrows():
            movie_features = movie[feature_columns].values.astype(np.float64)
            similarity = 1 / (1 + np.linalg.norm(search_features_flat - movie_features))
            similarities.append((movie['movieId'], movie['title'], similarity))
        
        # Ordenar y tomar top 10
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
        
        with col2:
            st.subheader(f"📋 Similares (Top {len(top_recommendations)})")
            
            if not top_recommendations:
                st.warning("No hay películas similares")
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

if __name__ == "__main__":
    main()