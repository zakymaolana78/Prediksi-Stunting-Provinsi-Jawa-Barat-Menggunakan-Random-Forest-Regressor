import streamlit as st
import pandas as pd
import ast
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

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
# CSS
# ==========================================================

st.markdown("""
<style>

.main{
    padding-top:20px;
}

[data-testid="stMetric"]{
    background:#f8f9fa;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# PATH
# ==========================================================

BASE_DIR = Path(__file__).parent

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_dataset():

    dataset = pd.read_csv(BASE_DIR / "dataset_final_stunting.csv")

    hasil = pd.read_csv(BASE_DIR / "hasil_prediksi_rf.csv")

    evaluasi = pd.read_csv(BASE_DIR / "hasil_evaluasi.csv")

    importance = pd.read_csv(BASE_DIR / "feature_importance.csv")

    best_parameter = pd.read_csv(BASE_DIR / "best_parameter_rf.csv")

    cross_validation = pd.read_csv(BASE_DIR / "cross_validation_rf.csv")

    tuning_result = pd.read_csv(BASE_DIR / "tuning_result_rf.csv")

    return (
        dataset,
        hasil,
        evaluasi,
        importance,
        best_parameter,
        cross_validation,
        tuning_result
    )

dataset, hasil, evaluasi, importance, best_parameter, cross_validation, tuning_result = load_dataset()

# ==========================================================
# MEMBANGUN MODEL RANDOM FOREST
# ==========================================================

X = dataset[
    [
        "persentase_penduduk_miskin",
        "garis_kemiskinan",
        "persentase_sanitasi_layak",
        "jumlah_tenaga_gizi"
    ]
]

y = dataset["jumlah_balita_stunting"]

# ==========================================================
# MEMBACA BEST PARAMETER
# ==========================================================

params = {}

for _, row in best_parameter.iterrows():

    parameter = str(row["Parameter"]).strip()
    nilai = row["Nilai"]

    # Jika kosong
    if pd.isna(nilai):
        params[parameter] = None
        continue

    # Jika berupa string
    if isinstance(nilai, str):

        nilai = nilai.strip()

        # None
        if nilai.lower() == "none":
            params[parameter] = None

        # Boolean
        elif nilai.lower() == "true":
            params[parameter] = True

        elif nilai.lower() == "false":
            params[parameter] = False

        # String khusus Random Forest
        elif nilai.lower() in ["sqrt", "log2"]:
            params[parameter] = nilai.lower()

        # Angka / List / Tuple
        else:

            try:
                params[parameter] = ast.literal_eval(nilai)

            except Exception:
                params[parameter] = nilai

    else:

        params[parameter] = nilai

# ==========================================================
# MEMASTIKAN RANDOM STATE
# ==========================================================

params["random_state"] = 42

# ==========================================================
# MEMBANGUN MODEL
# ==========================================================

model = RandomForestRegressor(**params)

model.fit(X, y)



# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/9/99/Coat_of_arms_of_West_Java.svg",
    width=90
)

st.sidebar.title("Prediksi Stunting")

menu = st.sidebar.radio(

    "Pilih Menu",

    [

        "🏠 Dashboard",

        "📂 Dataset",

        "📈 Visualisasi",

        "⚙️ Training Model",

        "🤖 Prediksi",

        "📉 Evaluasi",

        "⭐ Feature Importance",

        "📑 Analisis Model",

        "ℹ️ Tentang"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Random Forest Regressor

Data :

Open Data Jawa Barat

Periode :

2019 - 2024
"""
)

# ==========================================================
# DASHBOARD
# ==========================================================

if menu == "🏠 Dashboard":

    st.title("📊 Dashboard Prediksi Stunting Jawa Barat")

    st.write(
        """
Aplikasi ini digunakan untuk memprediksi jumlah balita stunting di Provinsi Jawa Barat menggunakan algoritma Random Forest Regressor berdasarkan faktor sosial ekonomi dan kesehatan masyarakat.
"""
    )

    r2 = evaluasi.loc[
        evaluasi["Metrik"]=="R2 Score",
        "Nilai"
    ].values[0]

    cv = evaluasi.loc[
        evaluasi["Metrik"]=="Cross Validation R2",
        "Nilai"
    ].values[0]

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Jumlah Data",
        len(dataset)
    )

    c2.metric(
        "Kabupaten/Kota",
        dataset["nama_kabupaten_kota"].nunique()
    )

    c3.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    c4.metric(
        "Cross Validation",
        f"{cv:.4f}"
    )

    st.divider()

    st.subheader("Preview Dataset")

    st.dataframe(
        dataset.head(10),
        use_container_width=True
    )

    st.divider()

    st.subheader("Statistik Dataset")

    st.dataframe(
        dataset.describe(),
        use_container_width=True
    )

    st.divider()

    st.subheader("Distribusi Data per Tahun")

    fig = px.histogram(
        dataset,
        x="tahun",
        color="tahun",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# ==========================================================
# DATASET
# ==========================================================

elif menu == "📂 Dataset":

    st.title("📂 Dataset Penelitian")

    st.write(
        """
Dataset yang digunakan merupakan hasil preprocessing dari beberapa dataset
Open Data Jawa Barat yang telah digabungkan menjadi satu dataset penelitian.
"""
    )

    st.metric(
        "Jumlah Data",
        len(dataset)
    )

    st.dataframe(
        dataset,
        use_container_width=True
    )

    csv = dataset.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Dataset",
        data=csv,
        file_name="dataset_final_stunting.csv",
        mime="text/csv"
    )

# ==========================================================
# VISUALISASI
# ==========================================================

elif menu == "📈 Visualisasi":

    st.title("📈 Visualisasi Dataset")

    pilihan = st.selectbox(

        "Pilih Jenis Visualisasi",

        [

            "Histogram",

            "Boxplot",

            "Scatter Plot",

            "Heatmap Korelasi"

        ]

    )

    # ======================================================
    # HISTOGRAM
    # ======================================================

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

            marginal="box",

            title=f"Histogram {kolom}"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ======================================================
    # BOXPLOT
    # ======================================================

    elif pilihan == "Boxplot":

        kolom = st.selectbox(

            "Pilih Kolom",

            [

                "jumlah_balita_stunting",

                "persentase_penduduk_miskin",

                "garis_kemiskinan",

                "persentase_sanitasi_layak",

                "jumlah_tenaga_gizi"

            ],

            key="boxplot"

        )

        fig = px.box(

            dataset,

            y=kolom,

            points="outliers",

            title=f"Boxplot {kolom}"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ======================================================
    # SCATTER
    # ======================================================

    elif pilihan == "Scatter Plot":

        fitur = [

            "jumlah_balita_stunting",

            "persentase_penduduk_miskin",

            "garis_kemiskinan",

            "persentase_sanitasi_layak",

            "jumlah_tenaga_gizi"

        ]

        x = st.selectbox(

            "Sumbu X",

            fitur

        )

        y = st.selectbox(

            "Sumbu Y",

            fitur,

            index=1

        )

        fig = px.scatter(

            dataset,

            x=x,

            y=y,

            color="tahun",

            hover_data=[

                "nama_kabupaten_kota"

            ],

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ======================================================
    # HEATMAP
    # ======================================================

    elif pilihan == "Heatmap Korelasi":

        corr = dataset.select_dtypes(

            include="number"

        ).corr()

        fig = px.imshow(

            corr,

            text_auto=True,

            color_continuous_scale="RdBu_r",

            aspect="auto",

            title="Heatmap Korelasi"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )
# ==========================================================
# TRAINING MODEL
# ==========================================================

elif menu == "⚙️ Training Model":

    st.title("⚙️ Training Model Random Forest")

    st.write("""
Model dibangun menggunakan algoritma **Random Forest Regressor**
dengan proses **RandomizedSearchCV** untuk mendapatkan hyperparameter
terbaik serta **5-Fold Cross Validation** untuk mengukur performa model.
""")

    st.subheader("Best Hyperparameter")

    st.dataframe(
        best_parameter,
        use_container_width=True
    )

    st.subheader("Hasil Cross Validation")

    st.dataframe(
        cross_validation,
        use_container_width=True
    )

    rata2 = cross_validation.iloc[-1]["R2 Score"]

    st.metric(
        "Rata-rata Cross Validation R²",
        f"{float(rata2):.4f}"
    )

    st.subheader("Seluruh Percobaan Hyperparameter")

    st.dataframe(
        tuning_result,
        use_container_width=True,
        height=350
    )

# ==========================================================
# PREDIKSI
# ==========================================================

elif menu == "🤖 Prediksi":

    st.title("🤖 Prediksi Jumlah Balita Stunting")

    st.write(
        "Masukkan nilai setiap variabel untuk melakukan prediksi."
    )

    col1, col2 = st.columns(2)

    with col1:

        miskin = st.number_input(
            "Persentase Penduduk Miskin",
            min_value=0.0,
            value=7.5,
            step=0.1
        )

        garis = st.number_input(
            "Garis Kemiskinan",
            min_value=0.0,
            value=450000.0,
            step=1000.0
        )

    with col2:

        sanitasi = st.number_input(
            "Persentase Sanitasi Layak",
            min_value=0.0,
            value=90.0,
            step=0.1
        )

        gizi = st.number_input(
            "Jumlah Tenaga Gizi",
            min_value=0.0,
            value=40.0,
            step=1.0
        )

    if st.button("Prediksi"):

        data = pd.DataFrame({

            "persentase_penduduk_miskin":[miskin],

            "garis_kemiskinan":[garis],

            "persentase_sanitasi_layak":[sanitasi],

            "jumlah_tenaga_gizi":[gizi]

        })

        with st.spinner("Sedang melakukan prediksi..."):

            hasil_prediksi = model.predict(data)[0]

        st.success(
            f"Prediksi Jumlah Balita Stunting : {hasil_prediksi:.0f} Balita"
        )

        st.subheader("Data Input")

        st.dataframe(
            data,
            use_container_width=True
        )

# ==========================================================
# EVALUASI MODEL
# ==========================================================

elif menu == "📉 Evaluasi":

    st.title("📉 Evaluasi Model")

    st.dataframe(
        evaluasi,
        use_container_width=True
    )

    st.subheader("Grafik Aktual vs Prediksi")

    fig = px.scatter(

        hasil,

        x="Aktual",

        y="Prediksi"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Residual Plot")

    fig = px.scatter(

        hasil,

        x="Prediksi",

        y="Residual"

    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Download Hasil Evaluasi")

    csv = evaluasi.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download Evaluasi",

        csv,

        "hasil_evaluasi.csv",

        "text/csv"

    )
# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

elif menu == "⭐ Feature Importance":

    st.title("⭐ Feature Importance")

    st.write("""
Feature Importance menunjukkan tingkat kontribusi setiap variabel
terhadap prediksi jumlah balita stunting yang dihasilkan oleh
algoritma Random Forest Regressor.
""")

    fig = px.bar(

        importance,

        x="Importance",

        y="Fitur",

        orientation="h",

        text="Importance",

        color="Importance",

        color_continuous_scale="Blues"

    )

    fig.update_layout(
        yaxis_title="Fitur",
        xaxis_title="Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Nilai Feature Importance")

    st.dataframe(
        importance,
        use_container_width=True
    )

    csv = importance.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download Feature Importance",

        csv,

        "feature_importance.csv",

        "text/csv"

    )

# ==========================================================
# ANALISIS MODEL
# ==========================================================

elif menu == "📑 Analisis Model":

    st.title("📑 Analisis Model")

    r2 = float(
        evaluasi.loc[
            evaluasi["Metrik"]=="R2 Score",
            "Nilai"
        ].values[0]
    )

    mae = float(
        evaluasi.loc[
            evaluasi["Metrik"]=="MAE",
            "Nilai"
        ].values[0]
    )

    rmse = float(
        evaluasi.loc[
            evaluasi["Metrik"]=="RMSE",
            "Nilai"
        ].values[0]
    )

    cv = float(
        evaluasi.loc[
            evaluasi["Metrik"]=="Cross Validation R2",
            "Nilai"
        ].values[0]
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    c2.metric(
        "MAE",
        f"{mae:.2f}"
    )

    c3.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    c4.metric(
        "Cross Validation",
        f"{cv:.4f}"
    )

    st.divider()

    st.subheader("Interpretasi")

    st.success(f"""
Model Random Forest memiliki nilai **R² sebesar {r2:.4f}**
atau sekitar **{r2*100:.2f}%**.

Hal ini menunjukkan bahwa model mampu menjelaskan sekitar
**{r2*100:.2f}% variasi jumlah balita stunting** berdasarkan
empat variabel input yang digunakan.
""")

    st.info(f"""
Nilai MAE sebesar **{mae:.2f}** menunjukkan rata-rata
kesalahan prediksi model terhadap data aktual.
""")

    st.info(f"""
Nilai RMSE sebesar **{rmse:.2f}** menunjukkan besarnya
kesalahan prediksi yang lebih sensitif terhadap error besar.
""")

    st.success(f"""
Hasil Cross Validation sebesar **{cv:.4f}**
menunjukkan bahwa model memiliki performa yang relatif stabil
pada setiap fold pengujian.
""")

    st.divider()

    st.subheader("Kesimpulan")

    st.write("""
Berdasarkan hasil evaluasi, algoritma Random Forest Regressor
mampu melakukan prediksi jumlah balita stunting dengan performa
yang baik. Nilai R² yang tinggi menunjukkan bahwa variabel
persentase penduduk miskin, garis kemiskinan,
persentase sanitasi layak, dan jumlah tenaga gizi
memberikan kontribusi terhadap prediksi jumlah balita stunting.
""")

# ==========================================================
# TENTANG
# ==========================================================

elif menu == "ℹ️ Tentang":

    st.title("ℹ️ Tentang Aplikasi")

    st.subheader("Judul Penelitian")

    st.write("""
**Penerapan Algoritma Random Forest Regressor
untuk Prediksi Jumlah Balita Stunting Berdasarkan
Faktor Sosial Ekonomi dan Kesehatan Masyarakat
di Provinsi Jawa Barat**
""")

    st.divider()

    st.subheader("Dataset")

    st.write("""
Dataset berasal dari **Open Data Jawa Barat**
dengan periode **2019–2024**.

Variabel yang digunakan meliputi:

- Jumlah Balita Stunting
- Persentase Penduduk Miskin
- Garis Kemiskinan
- Persentase Sanitasi Layak
- Jumlah Tenaga Kesehatan Gizi
""")

    st.divider()

    st.subheader("Metode")

    st.write("""
Metodologi penelitian menggunakan **CRISP-DM** yang terdiri dari:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment
""")

    st.divider()

    st.subheader("Algoritma")

    st.write("""
Model prediksi dibangun menggunakan:

- Random Forest Regressor
- RandomizedSearchCV
- 5-Fold Cross Validation
- Feature Importance
""")

    st.divider()

    st.subheader("Tools")

    st.write("""
- Python
- Scikit-Learn
- Pandas
- Plotly
- Streamlit
""")

    st.divider()

    st.caption("© 2026 | Aplikasi Prediksi Stunting Jawa Barat")
