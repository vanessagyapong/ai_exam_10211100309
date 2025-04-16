#Vanessa Gyapong
#10211100309


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.set_page_config(page_title="ML & AI Explorer", layout="wide")
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a task:", [
    "Regression",
    "Clustering",
    "Neural Network",
    "Large Language Model (LLM)"
])

st.title("Machine Learning & AI Exploration App")

if app_mode == "Regression":
    st.header("Regression Task")
    uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of dataset:", df.head())
        
        # Store the original version of the dataset (before encoding and standardization)
        original_df = df.copy()

        # Handle Missing Values Column-wise
        st.subheader("Handle Missing Values Column-wise")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                strategy = st.selectbox(f"Handle nulls in '{col}'", 
                                        ["None", "Drop rows", "Mean", "Median", "Mode"], key=col)
                if strategy == "Drop rows":
                    df = df.dropna(subset=[col])
                elif strategy == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)

        st.write("Handled Missing Data Dataset Preview:", df.head())

        # Preprocessing categorical columns (string to numeric)
        st.subheader("Preprocess Categorical Data")
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Save encoder for inverse transformation if needed

        st.write("Processed Dataset Preview:", df.head())

        target = st.text_input("Enter target column name")

        if target and target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]

            if st.checkbox("Standardize data"):
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            test_size = st.slider("Test size (for evaluation)", 0.0, 0.1, 0.5, 0.2)
            if test_size==0.0:
                test_size= None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("MAE:", mean_absolute_error(y_test, y_pred))
            st.write("R² Score:", r2_score(y_test, y_pred))

            # Scatter plot for actual vs predicted
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            # Feature vs Actual plot with regression line
            feature = st.selectbox("Choose a feature to plot against", X.columns)
            fig, ax = plt.subplots()
            ax.scatter(X_test[feature], y_test, color="blue", label="Data Points")
            ax.plot(X_test[feature], model.predict(X_test), color="red", linestyle="--", label="Regression Line")
            ax.set_xlabel(feature)
            ax.set_ylabel("Actual Values")
            ax.set_title(f"Regression Line for {feature} vs Actual")
            ax.legend()
            st.pyplot(fig)

            # Custom Prediction Input (excluding target column)
            st.subheader("Custom Prediction")
            input_data = {}
            for col in original_df.columns:
                if col != target:  # Exclude the target column
                    # If the column is categorical, use a selectbox for original string values
                    if original_df[col].dtype == 'object':
                        input_data[col] = st.selectbox(f"{col}", options=original_df[col].unique())
                    else:
                        input_data[col] = st.number_input(f"{col}")

            if st.button("Predict"):
                # Convert the input data into a DataFrame
                input_df = pd.DataFrame([input_data])

                # Apply label encoding for categorical columns
                for col in input_df.select_dtypes(include=['object']).columns:
                    if col in label_encoders:
                        input_df[col] = label_encoders[col].transform(input_df[col])

                # Apply standardization if needed
                if 'scaler' in locals():
                    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

                st.write("Prediction:", model.predict(input_df)[0])


elif app_mode == "Clustering":
    st.header("Clustering Task")
    uploaded_file = st.file_uploader("Upload CSV dataset for clustering", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 📋 Dataset Preview", df.head())

        # Handle missing values column-wise
        st.subheader("Handle Missing Values Column-wise")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                strategy = st.selectbox(f"Handle nulls in '{col}'",
                                        ["None", "Drop rows", "Mean", "Median", "Mode"], key=col)
                if strategy == "Drop rows":
                    df = df.dropna(subset=[col])
                elif strategy == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
        st.write("Handled Missing Data Dataset Preview:", df.head())

         # Preprocessing categorical columns (string to numeric)
        st.subheader("Preprocess Categorical Data")
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Save encoder for inverse transformation if needed

        st.write("Processed Dataset Preview:", df.head())   

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Dataset must contain at least two numeric features for clustering.")
        else:
            st.subheader("Clustering Options")
            features = st.multiselect("Select features to cluster on", numeric_cols, default=numeric_cols[:2])

            if len(features) >= 2:
                X = df[features]

                # Normalize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Elbow method to auto-suggest K
                st.subheader("Select Number of Clusters (K)")
                if st.button("🔍 Use Elbow Method to Suggest K"):
                    distortions = []
                    K_range = range(1, 11)
                    for k in K_range:
                        km = KMeans(n_clusters=k, random_state=42)
                        km.fit(X_scaled)
                        distortions.append(km.inertia_)

                    # Plot elbow
                    fig_elbow, ax_elbow = plt.subplots()
                    ax_elbow.plot(K_range, distortions, 'bo-')
                    ax_elbow.set_xlabel('Number of clusters (k)')
                    ax_elbow.set_ylabel('Inertia (Distortion)')
                    ax_elbow.set_title('Elbow Method For Optimal k')
                    st.pyplot(fig_elbow)

                    # Estimate optimal k using 2nd derivative
                    deltas = np.diff(distortions, 2)
                    if len(deltas) > 0:
                        suggested_k = np.argmin(deltas) + 2
                        st.success(f"Optimal K suggested by Elbow Method: **{suggested_k}**")
                    else:
                        suggested_k = 3
                        st.warning("Unable to determine elbow point automatically. Defaulting to k=3.")
                else:
                    suggested_k = 3

                n_clusters = st.slider("Select number of clusters", 2, 10, suggested_k)

                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                df['Cluster'] = cluster_labels

                st.write("### 📊 Clustered Dataset", df.head())

                # 2D Visualization
                if len(features) == 2:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=X[features[0]], y=X[features[1]], hue=cluster_labels, palette='Set2', ax=ax)
                    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5, marker='X', label='Centroids')
                    ax.set_title("2D Cluster Plot")
                    ax.legend()
                    st.pyplot(fig)

                # 3D Visualization
                elif len(features) >= 3:
                    pca = PCA(n_components=3)
                    X_pca = pca.fit_transform(X_scaled)
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cluster_labels, cmap='Set2')
                    ax.set_title("3D Cluster Plot (PCA reduced)")
                    st.pyplot(fig)

                # Download clustered dataset
                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Clustered Dataset", data=csv, file_name="clustered_data.csv", mime='text/csv')

            else:
                st.warning("Please select at least two features for clustering.")


elif app_mode == "Neural Network":


    st.header("Neural Network Task")
    uploaded_file = st.file_uploader("Upload dataset for classification (CSV)", type="csv")
    
    if uploaded_file:
      



        df = pd.read_csv(uploaded_file)
        st.write("📊 Dataset Preview", df.head())

        # 2. Target column selection
        target_col = st.text_input("🎯 Enter target column name")
        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Encode categorical target labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            num_classes = len(np.unique(y_encoded))

            # Encode categorical features in X
            categorical_encoders = {}
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    categorical_encoders[col] = le  # Save encoder for prediction

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train-validation split
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

            # Convert labels to categorical (one-hot)
            y_train_cat = to_categorical(y_train, num_classes)
            y_val_cat = to_categorical(y_val, num_classes)

            # 3. Hyperparameter tuning
            st.sidebar.header("⚙️ Hyperparameters")
            learning_rate = st.sidebar.number_input("Learning Rate", value=0.001)
            epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, value=20)
            hidden_units = st.sidebar.slider("Hidden Units", min_value=16, max_value=256, value=64)

            model = Sequential([
                Dense(hidden_units, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(hidden_units, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            if st.button("🚀 Train Model"):
                with st.spinner("Training..."):
                    history = model.fit(
                        X_train, y_train_cat,
                        validation_data=(X_val, y_val_cat),
                        epochs=epochs,
                        verbose=0
                    )
                st.success("✅ Training completed!")
                st.write("Classes:", label_encoder.classes_)


                # Save everything to session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.label_encoder = label_encoder
                st.session_state.categorical_encoders = categorical_encoders
                st.session_state.columns = list(X.columns)

                # Plot training history
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(history.history['loss'], label='Train Loss')
                ax[0].plot(history.history['val_loss'], label='Val Loss')
                ax[0].set_title("Loss over Epochs")
                ax[0].legend()

                ax[1].plot(history.history['accuracy'], label='Train Accuracy')
                ax[1].plot(history.history['val_accuracy'], label='Val Accuracy')
                ax[1].set_title("Accuracy over Epochs")
                ax[1].legend()

                st.pyplot(fig)

            if 'model' in st.session_state:
                st.subheader("🔍 Make Prediction on New Input")

                input_data = {}
                for col in st.session_state.columns:
                    val = st.text_input(f"Enter value for '{col}'")
                    input_data[col] = val

                if st.button("📈 Predict"):
                    input_df = pd.DataFrame([input_data])

                    # Encode input using saved encoders
                    for col in input_df.columns:
                        if col in st.session_state.categorical_encoders:
                            le = st.session_state.categorical_encoders[col]
                            try:
                                input_df[col] = le.transform([input_df[col][0]])
                            except ValueError:
                                st.error(f"⚠️ Value '{input_df[col][0]}' not seen during training in column '{col}'")
                                st.stop()
                        else:
                            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

                    if input_df.isnull().any().any():
                        st.error("⚠️ Invalid input detected. Please check your entries.")
                    else:
                        input_scaled = st.session_state.scaler.transform(input_df)
                        prediction = st.session_state.model.predict(input_scaled)
                        predicted_class = st.session_state.label_encoder.inverse_transform([np.argmax(prediction)])
                        st.success(f"🧠 Predicted Class: **{predicted_class[0]}**")



elif app_mode == "Large Language Model (LLM)":
    load_dotenv()
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "handbook.pdf")


    # Initialize Qdrant client
    qdrant = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)

    # Load embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_text_from_pdf(pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def split_text(text, chunk_size=700, chunk_overlap=100):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)
    
    def retrieve_relevant_chunks(query):
        query_embedding = embedding_model.encode([query])[0]
        search_results = qdrant.search(
        collection_name="handbook",
        query_vector=query_embedding.tolist(),
        limit=10
    )
        return [hit.payload["text"] for hit in search_results]

# Function to generate answer using the Hugging Face model
    def generate_answer(query, retrieved_text):
        prompt = (
        "You are an AI assistant that answers questions based on provided document excerpts. "
        "Your goal is to extract the most relevant information and provide a concise, factual summary.\n\n"
        "### Document Excerpts:\n"
        f"{retrieved_text}\n\n"
        "### User Question:\n"
        f"{query}\n\n"
        "### Instructions:\n"
        "- Identify the key points that directly answer the user's question.\n"
        "- Provide a **clear and structured summary** in **2-3 sentences**.\n"
        "- Avoid unnecessary details or repeating the text verbatim.\n"
        "- If the excerpts do not contain enough information, state: 'The document does not provide a clear answer to this question.'\n\n"
        "### Answer:"
    )

        response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-large",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={
            "inputs": prompt,
            "parameters": {
                "temperature": 0.3,  # Low temp for accuracy
                "max_new_tokens": 150  # Controls response length
            }
        }
    )
  
        generated_text = response.json()[0]["generated_text"]
    
    # Split by "### Answer:" to isolate the actual response
        answer = generated_text.split("### Answer:")[-1].strip()
    
        return answer


    st.header("Large Language Model Q&A")


    # Extract text from the PDF and process
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text(pdf_text)

    # Create embeddings for the document chunks and upload to Qdrant
    chunk_embeddings = np.array(embedding_model.encode(text_chunks))
    qdrant.recreate_collection(
        collection_name="handbook",
        vectors_config=VectorParams(
            size=chunk_embeddings.shape[1],
            distance=Distance.COSINE
        ),
    )

    # Upload embeddings to Qdrant
    qdrant.upload_points(
        collection_name="handbook",
        points=[
            PointStruct(id=i, vector=chunk_embeddings[i].tolist(), payload={"text": text_chunks[i]})
            for i in range(len(text_chunks))
        ]
    )

    st.subheader("Ask a Question")
    query = st.text_input("Enter your question")
    if st.button("Ask") and query:
        relevant_chunks = retrieve_relevant_chunks(query)
        rag_answer = generate_answer(query, "\n".join(relevant_chunks))
        st.write("Answer:", rag_answer)