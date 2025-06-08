import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Rekomendasi Penjurusan SMA",
    page_icon="🎓",
    layout="wide"
)

# CSS sederhana
st.markdown("""
<style>
.big-font {
    font-size: 24px !important;
    font-weight: bold;
    color: #1f77b4;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #f0f8ff;
    border: 2px solid #1f77b4;
    text-align: center;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎓 Sistem Rekomendasi Penjurusan SMA")
st.write("Masukkan nilai siswa untuk mendapatkan rekomendasi jurusan: **IPA**, **IPS**, atau **Bahasa**")
st.markdown("---")

# Fungsi untuk menghitung fitur engineered
def compute_engineered_features(df):
    """Menghitung fitur Mean dan Trend untuk setiap mata pelajaran"""
    subjects = ['Agama', 'PKN', 'Indo', 'Mate', 'IPA', 'IPS', 'Inggris', 'Senbud', 'PJOK', 'Prakarya', 'B_Daerah']
    
    engineered = pd.DataFrame(index=df.index)
    
    for subj in subjects:
        # Kolom semester untuk mata pelajaran ini
        sem_cols = [f"{subj}{i}" for i in range(1, 7)]
        
        # Pastikan semua kolom ada
        for col in sem_cols:
            if col not in df.columns:
                df[col] = 75  # nilai default
        
        # Hitung Mean
        engineered[f"{subj}_Mean"] = df[sem_cols].mean(axis=1)
        
        # Hitung Trend (slope dari semester 1-6)
        trends = []
        for idx in df.index:
            values = df.loc[idx, sem_cols].values
            # Hitung slope menggunakan least squares
            x = np.arange(1, 7)  # semester 1-6
            y = values
            slope = np.polyfit(x, y, 1)[0]
            trends.append(slope)
        
        engineered[f"{subj}_Trend"] = trends
    
    return engineered

# Buat model dummy yang simple
@st.cache_resource
def create_simple_model():
    """Buat model dummy sederhana"""
    np.random.seed(42)
    
    # Data dummy untuk training (engineered features)
    subjects = ['Agama', 'PKN', 'Indo', 'Mate', 'IPA', 'IPS', 'Inggris', 'Senbud', 'PJOK', 'Prakarya', 'B_Daerah']
    
    # Buat 1000 sample dummy
    data = []
    labels = []
    
    for i in range(1000):
        row = {}
        # Generate mean dan trend untuk setiap subject
        for subj in subjects:
            mean_val = np.random.normal(75, 10)
            trend_val = np.random.normal(0, 2)
            row[f"{subj}_Mean"] = max(0, min(100, mean_val))
            row[f"{subj}_Trend"] = trend_val
        
        data.append(row)
        
        # Simple logic untuk label
        if row['Mate_Mean'] > 80 or row['IPA_Mean'] > 80:
            labels.append('IPA')
        elif row['IPS_Mean'] > 80:
            labels.append('IPS')
        else:
            labels.append('Bahasa')
    
    # Train model
    X = pd.DataFrame(data)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, labels)
    
    return model, scaler

# Load model dengan error handling
@st.cache_resource
def load_model():
    try:
        if os.path.exists("random_forest_model.pkl") and os.path.exists("scaler.pkl"):
            model = joblib.load("random_forest_model.pkl")
            scaler = joblib.load("scaler.pkl")
            st.sidebar.success("✅ Model asli berhasil dimuat")
            return model, scaler, False
        else:
            model, scaler = create_simple_model()
            st.sidebar.warning("⚠️ Menggunakan model demo")
            return model, scaler, True
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        model, scaler = create_simple_model()
        st.sidebar.warning("⚠️ Fallback ke model demo")
        return model, scaler, True

# Load model
model, scaler, is_demo = load_model()

# Sidebar info
with st.sidebar:
    st.header("📚 Mata Pelajaran")
    subjects_info = [
        "1. Pendidikan Agama",
        "2. PKN", 
        "3. Bahasa Indonesia",
        "4. Matematika",
        "5. IPA",
        "6. IPS", 
        "7. Bahasa Inggris",
        "8. Seni Budaya",
        "9. PJOK",
        "10. Prakarya",
        "11. Bahasa Daerah"
    ]
    
    for info in subjects_info:
        st.write(info)

# Input form
st.subheader("📝 Input Nilai Siswa")

# Daftar mata pelajaran
subjects = ['Agama', 'PKN', 'Indo', 'Mate', 'IPA', 'IPS', 'Inggris', 'Senbud', 'PJOK', 'Prakarya', 'B_Daerah']
subject_names = ['Pendidikan Agama', 'PKN', 'Bahasa Indonesia', 'Matematika', 'IPA', 'IPS', 'Bahasa Inggris', 'Seni Budaya', 'PJOK', 'Prakarya', 'Bahasa Daerah']

# Buat tabs untuk setiap mata pelajaran
tabs = st.tabs(subject_names)

input_data = {}

for i, (tab, subj, name) in enumerate(zip(tabs, subjects, subject_names)):
    with tab:
        st.write(f"**Masukkan nilai {name} untuk semester 1-6:**")
        cols = st.columns(6)
        
        for sem in range(1, 7):
            with cols[sem-1]:
                key = f"{subj}{sem}"
                input_data[key] = st.number_input(
                    f"Sem {sem}",
                    min_value=0,
                    max_value=100,
                    value=75,
                    key=key
                )

# Tombol prediksi
st.markdown("---")
if st.button("🔍 Prediksi Jurusan", type="primary", use_container_width=True):
    try:
        # Buat DataFrame dari input
        df_input = pd.DataFrame([input_data])
        
        # Hitung engineered features
        engineered_features = compute_engineered_features(df_input)
        
        # Preprocessing
        X_scaled = scaler.transform(engineered_features)
        
        # Prediksi
        prediction = model.predict(X_scaled)[0]
        
        # Tampilkan hasil
        st.markdown(f"""
        <div class="result-box">
            <h2>🎯 Hasil Prediksi</h2>
            <h1 style="color: #1f77b4; font-size: 48px;">{prediction}</h1>
            <p>Jurusan yang direkomendasikan untuk siswa ini</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tampilkan probabilitas jika tersedia
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)[0]
            classes = model.classes_
            
            st.subheader("📊 Tingkat Kepercayaan:")
            
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            for i, (kelas, prob) in enumerate(zip(classes, proba)):
                with cols[i]:
                    st.metric(
                        label=kelas,
                        value=f"{prob:.1%}",
                        delta=None
                    )
                    st.progress(prob)
        
        # Analisis nilai
        st.subheader("📈 Analisis Nilai")
        
        # Hitung rata-rata per mata pelajaran
        subject_averages = {}
        for subj, name in zip(subjects, subject_names):
            semester_values = [input_data[f"{subj}{sem}"] for sem in range(1, 7)]
            avg = np.mean(semester_values)
            subject_averages[name] = avg
        
        # Tampilkan dalam bentuk tabel
        analysis_df = pd.DataFrame(list(subject_averages.items()), 
                                 columns=['Mata Pelajaran', 'Rata-rata'])
        analysis_df = analysis_df.sort_values('Rata-rata', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Rata-rata Nilai per Mata Pelajaran:**")
            st.dataframe(analysis_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**🏆 3 Nilai Tertinggi:**")
            top_3 = analysis_df.head(3)
            for idx, row in top_3.iterrows():
                st.write(f"• **{row['Mata Pelajaran']}**: {row['Rata-rata']:.1f}")
            
            st.write("**📝 3 Nilai Terendah:**")
            bottom_3 = analysis_df.tail(3)
            for idx, row in bottom_3.iterrows():
                st.write(f"• **{row['Mata Pelajaran']}**: {row['Rata-rata']:.1f}")
        
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")
        st.write("**Debug info:**")
        st.write(f"Input data shape: {len(input_data)}")
        st.write(f"Model type: {type(model)}")

# Upload file section
st.markdown("---")
st.subheader("📁 Upload File CSV")

uploaded_file = st.file_uploader("Upload file CSV dengan data siswa", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File berhasil diupload!")
        
        # Preview
        st.write("**Preview data:**")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("🚀 Prediksi Batch", type="secondary"):
            with st.spinner("Memproses..."):
                # Hitung engineered features
                engineered_df = compute_engineered_features(df)
                
                # Prediksi
                X_scaled = scaler.transform(engineered_df)
                predictions = model.predict(X_scaled)
                
                # Tambahkan hasil ke dataframe
                result_df = df.copy()
                result_df['Prediksi_Jurusan'] = predictions
                
                # Tampilkan hasil
                st.success("✅ Prediksi batch selesai!")
                st.dataframe(result_df, use_container_width=True)
                
                # Statistik
                jurusan_counts = pd.Series(predictions).value_counts()
                st.write("**Distribusi Prediksi:**")
                
                col1, col2, col3 = st.columns(3)
                cols = [col1, col2, col3]
                
                for i, (jurusan, count) in enumerate(jurusan_counts.items()):
                    if i < 3:
                        with cols[i]:
                            percentage = (count / len(predictions)) * 100
                            st.metric(jurusan, f"{count} siswa", f"{percentage:.1f}%")
                
                # Download
                csv_data = result_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Hasil",
                    data=csv_data,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"❌ Error memproses file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎓 Sistem Rekomendasi Penjurusan SMA</p>
    <p><small>Dikembangkan untuk membantu siswa memilih jurusan yang tepat</small></p>
</div>
""", unsafe_allow_html=True)
