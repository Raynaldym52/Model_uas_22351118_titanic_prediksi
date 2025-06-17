import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

st.title("Analisis Dataset Titanic")

st.markdown("Upload file **Titanic-Dataset.csv** untuk mulai analisis.")

uploaded_file = st.file_uploader("Pilih file Titanic-Dataset.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Awal")
    st.write(df.head())

    st.subheader("Informasi Dataset")
    buffer = []
    df.info(buf=buffer.append)
    st.text("\n".join(buffer))

    # Visualisasi
    st.subheader("Distribusi Penumpang yang Selamat")
    fig1, ax1 = plt.subplots()
    df['Survived'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Jumlah Penumpang Selamat (1) dan Tidak (0)')
    st.pyplot(fig1)

    st.subheader("Distribusi Jenis Kelamin")
    fig2, ax2 = plt.subplots()
    df['Sex'].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_title('Distribusi Jenis Kelamin')
    st.pyplot(fig2)

    # Pra-pemrosesan
    df = df.dropna()
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    y = df['Survived']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, verbose=0)

    st.subheader("Evaluasi Model")
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)
else:
    st.info("Silakan upload file CSV terlebih dahulu.")
