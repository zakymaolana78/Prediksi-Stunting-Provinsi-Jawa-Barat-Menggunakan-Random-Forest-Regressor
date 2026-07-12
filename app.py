import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Prediksi Stunting Jawa Barat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD DATA
# ==========================================================

BASE_DIR = Path(__file__).parent

dataset = pd.read_csv(BASE_DIR / "dataset_final_stunting.csv")
hasil = pd.read_csv(BASE_DIR / "hasil_prediksi_rf.csv")
evaluasi = pd.read_csv(BASE_DIR / "hasil_evaluasi.csv")
importance = pd.read_csv(BASE_DIR / "feature_importance.csv")

model = joblib.load(BASE_DIR / "model_stunting_rf.joblib")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📊 Menu")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Home",
        "📂 Dataset",
        "📈 Visualisasi",
        "🤖 Prediksi",
        "📉 Evaluasi",
        "⭐ Feature Importance",
        "ℹ️ Tentang"
    ]
)

# ==========================================================
# HOME
# ==========================================================

if menu == "🏠 Home":

    st.title("Prediksi Jumlah Balita Stunting")
    st.subheader("Provinsi Jawa Barat")
    st.write(
        """
        Aplikasi ini digunakan untuk melakukan prediksi jumlah balita
        stunting menggunakan algoritma **Random Forest Regressor**.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Jumlah Data", len(dataset))
    c2.metric("Jumlah Fitur", 4)
    c3.metric("Jumlah Kabupaten/Kota", dataset["nama_kabupaten_kota"].nunique())
    c4.metric("Rentang Tahun", "2019-2024")

    st.markdown("---")

    st.subheader("Preview Dataset")

    st.dataframe(dataset.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("Statistik Dataset")

    st.dataframe(dataset.describe(), use_container_width=True)

# ==========================================================
# DATASET
# ==========================================================

elif menu == "📂 Dataset":

    st.title("Dataset Final")

    st.write(
        "Dataset hasil preprocessing yang digunakan dalam penelitian."
    )

    st.dataframe(dataset, use_container_width=True)

    csv = dataset.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Dataset",
        csv,
        "dataset_final_stunting.csv",
        "text/csv"
    )

# ==========================================================
# VISUALISASI
# ==========================================================

elif menu == "📈 Visualisasi":

    st.title("Visualisasi Data")

    pilihan = st.selectbox(
        "Pilih Visualisasi",
        [
            "Histogram",
            "Scatter Plot",
            "Korelasi"
        ]
    )

    if pilihan == "Histogram":

        kolom = st.selectbox(
            "Pilih Kolom",
            [
                "jumlah_balita_stunting",
                "persentase_penduduk_miskin",
                "garis_kemiskinan",
                "persentase_sanitasi_layak",
                "jumlah_tenaga_gizi"
            ]
        )

        fig = px.histogram(
            dataset,
            x=kolom,
            nbins=25,
            title=f"Histogram {kolom}"
        )

        st.plotly_chart(fig, use_container_width=True)

    elif pilihan == "Scatter Plot":

        x = st.selectbox(
            "Sumbu X",
            dataset.columns[3:]
        )

        y = st.selectbox(
            "Sumbu Y",
            dataset.columns[3:],
            index=1
        )

        fig = px.scatter(
            dataset,
            x=x,
            y=y,
            color="tahun",
            hover_data=["nama_kabupaten_kota"]
        )

        st.plotly_chart(fig, use_container_width=True)

    elif pilihan == "Korelasi":

        corr = dataset.select_dtypes(include="number").corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )

        st.plotly_chart(fig, use_container_width=True)
      # ==========================================================
# PREDIKSI
# ==========================================================

elif menu == "🤖 Prediksi":

    st.title("Prediksi Jumlah Balita Stunting")

    st.write(
        "Masukkan nilai setiap variabel untuk memperoleh hasil prediksi."
    )

    col1, col2 = st.columns(2)

    with col1:

        penduduk_miskin = st.number_input(
            "Persentase Penduduk Miskin",
            min_value=0.0,
            max_value=100.0,
            value=7.50,
            step=0.01
        )

        garis_kemiskinan = st.number_input(
            "Garis Kemiskinan",
            min_value=100000.0,
            value=500000.0,
            step=1000.0
        )

    with col2:

        sanitasi = st.number_input(
            "Persentase Sanitasi Layak",
            min_value=0.0,
            max_value=100.0,
            value=95.00,
            step=0.01
        )

        tenaga_gizi = st.number_input(
            "Jumlah Tenaga Gizi",
            min_value=0,
            value=30,
            step=1
        )

    if st.button("Prediksi"):

        data = pd.DataFrame({

            "persentase_penduduk_miskin":[penduduk_miskin],

            "garis_kemiskinan":[garis_kemiskinan],

            "persentase_sanitasi_layak":[sanitasi],

            "jumlah_tenaga_gizi":[tenaga_gizi]

        })

        hasil_prediksi = model.predict(data)[0]

        st.success(
            f"Perkiraan Jumlah Balita Stunting : **{hasil_prediksi:,.0f} Balita**"
        )

        st.dataframe(data)

# ==========================================================
# EVALUASI
# ==========================================================

elif menu == "📉 Evaluasi":

    st.title("Evaluasi Model")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MAE",
        f"{evaluasi.loc[evaluasi['Metrik']=='MAE','Nilai'].values[0]:,.2f}"
    )

    c2.metric(
        "MSE",
        f"{evaluasi.loc[evaluasi['Metrik']=='MSE','Nilai'].values[0]:,.2f}"
    )

    c3.metric(
        "RMSE",
        f"{evaluasi.loc[evaluasi['Metrik']=='RMSE','Nilai'].values[0]:,.2f}"
    )

    c4.metric(
        "R²",
        f"{evaluasi.loc[evaluasi['Metrik']=='R2 Score','Nilai'].values[0]:.4f}"
    )

    st.markdown("---")

    st.subheader("Hasil Prediksi")

    st.dataframe(
        hasil,
        use_container_width=True
    )

    fig = px.scatter(

        hasil,

        x="Aktual",

        y="Prediksi",

        color="Residual",

        title="Aktual vs Prediksi"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.scatter(

        hasil,

        x="Prediksi",

        y="Residual",

        title="Residual Plot"

    )

    fig2.add_hline(
        y=0,
        line_dash="dash",
        line_color="red"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

elif menu == "⭐ Feature Importance":

    st.title("Feature Importance")

    st.dataframe(
        importance,
        use_container_width=True
    )

    fig = px.bar(

        importance,

        x="Importance",

        y="Fitur",

        orientation="h",

        text="Importance",

        title="Feature Importance Random Forest"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
  # ==========================================================
# TENTANG
# ==========================================================

elif menu == "ℹ️ Tentang":

    st.title("Tentang Aplikasi")

    st.markdown("""
    ## Prediksi Jumlah Balita Stunting di Provinsi Jawa Barat

    **Judul Skripsi**

    Penerapan Algoritma Random Forest Regressor untuk Prediksi Jumlah Balita
    Stunting Berdasarkan Faktor Sosial Ekonomi dan Kesehatan Masyarakat
    di Provinsi Jawa Barat.

    ---
    """)

    c1, c2 = st.columns(2)

    with c1:

        st.info("""
        **Dataset**

        - Jumlah Balita Stunting
        - Persentase Penduduk Miskin
        - Garis Kemiskinan
        - Persentase Sanitasi Layak
        - Jumlah Tenaga Gizi

        Sumber:
        Open Data Provinsi Jawa Barat
        """)

    with c2:

        st.success("""
        **Algoritma**

        Random Forest Regressor

        Library

        - Scikit-learn
        - Pandas
        - Plotly
        - Streamlit
        """)

    st.markdown("---")

    st.subheader("Informasi Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Data",
        len(dataset)
    )

    col2.metric(
        "Jumlah Kabupaten/Kota",
        dataset["nama_kabupaten_kota"].nunique()
    )

    col3.metric(
        "Periode",
        "2019 - 2024"
    )

# ==========================================================
# DASHBOARD TAMBAHAN
# ==========================================================

st.markdown("---")

st.subheader("Ringkasan Dataset")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Minimum Stunting",
    int(dataset["jumlah_balita_stunting"].min())
)

c2.metric(
    "Rata-rata",
    round(dataset["jumlah_balita_stunting"].mean(),2)
)

c3.metric(
    "Maksimum",
    int(dataset["jumlah_balita_stunting"].max())
)

# ==========================================================
# GRAFIK TREN
# ==========================================================

st.markdown("---")

st.subheader("Tren Jumlah Balita Stunting")

trend = dataset.groupby("tahun")["jumlah_balita_stunting"].sum().reset_index()

fig = px.line(

    trend,

    x="tahun",

    y="jumlah_balita_stunting",

    markers=True,

    title="Perkembangan Jumlah Balita Stunting"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================================
# TOP 10 KABUPATEN/KOTA
# ==========================================================

st.markdown("---")

st.subheader("Top 10 Kabupaten/Kota")

ranking = (

    dataset

    .groupby("nama_kabupaten_kota")["jumlah_balita_stunting"]

    .mean()

    .sort_values(

        ascending=False

    )

    .head(10)

    .reset_index()

)

fig = px.bar(

    ranking,

    x="jumlah_balita_stunting",

    y="nama_kabupaten_kota",

    orientation="h",

    title="Rata-rata Jumlah Balita Stunting"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# ==========================================================
# DOWNLOAD HASIL PREDIKSI
# ==========================================================

st.markdown("---")

csv = hasil.to_csv(index=False).encode("utf-8")

st.download_button(

    "📥 Download Hasil Prediksi",

    csv,

    "hasil_prediksi_rf.csv",

    "text/csv"

)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

### Prediksi Stunting Provinsi Jawa Barat

Menggunakan Random Forest Regressor

Dibangun menggunakan ❤️ Streamlit

© 2026

</div>
""",
unsafe_allow_html=True
)
