import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 
 
# Judul 
st.title("🎓 Sistem Rekomendasi Penjurusan SMA") 
st.write("Gunakan data nilai akademik siswa untuk memprediksi jurusan: **IPA**, **IPS**, atau **Bahasa**.") 
 
# Load model dan preprocessing 
try:
    model = joblib.load("random_forest_model.pkl") 
    scaler = joblib.load("scaler.pkl") 
    st.success("✅ Model dan scaler berhasil dimuat!")
except FileNotFoundError as e:
    st.error(f"❌ File model tidak ditemukan: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error saat memuat model: {e}")
    st.stop()
 
# Pilihan metode input 
option = st.radio("Pilih metode input data:", ["Input Manual", "Input dari URL"]) 
 
# ---------------------------------------- 
# 🔹 Opsi 1: Input Manual 
# ---------------------------------------- 
if option == "Input Manual": 
    st.subheader("Masukkan Nilai Semester (1-6) untuk Setiap Mata Pelajaran") 
 
    pelajaran = ['Agama', 'PKN', 'Indo', 'Mate', 'IPA', 'IPS',  
                 'Inggris', 'Senbud', 'PJOK', 'Prakarya', 'B_Daerah'] 
     
    input_data = {} 
    
    # Membuat layout kolom untuk input yang lebih rapi
    col1, col2 = st.columns(2)
    
    for i, pel in enumerate(pelajaran):
        if i % 2 == 0:
            with col1:
                st.write(f"**{pel}**")
                for sem in range(1, 7): 
                    kolom = f"{pel}{sem}" 
                    input_data[kolom] = st.number_input(f"Semester {sem}", 0, 100, 75, key=kolom)
        else:
            with col2:
                st.write(f"**{pel}**")
                for sem in range(1, 7): 
                    kolom = f"{pel}{sem}" 
                    input_data[kolom] = st.number_input(f"Semester {sem}", 0, 100, 75, key=kolom)
 
    if st.button("🔍 Prediksi Jurusan"): 
        try:
            df_input = pd.DataFrame([input_data]) 
            df_scaled = scaler.transform(df_input) 
            pred = model.predict(df_scaled) 
            
            # Karena tidak menggunakan label encoder, langsung ambil hasil prediksi
            jurusan = pred[0]  # Asumsi model sudah mengembalikan string langsung
            
            st.success(f"✅ Jurusan yang Direkomendasikan: **{jurusan}**") 
            
            # Menampilkan probabilitas jika model mendukung predict_proba
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df_scaled)[0]
                classes = model.classes_
                
                st.subheader("📊 Tingkat Kepercayaan Prediksi:")
                for i, (kelas, prob) in enumerate(zip(classes, proba)):
                    percentage = prob * 100
                    st.write(f"**{kelas}**: {percentage:.1f}%")
                    st.progress(prob)
                    
        except Exception as e:
            st.error(f"❌ Error saat melakukan prediksi: {e}")
 
# ---------------------------------------- 
# 🔹 Opsi 2: Input dari URL 
# ---------------------------------------- 
elif option == "Input dari URL": 
    st.subheader("Masukkan URL ke File CSV atau Excel") 
 
    file_url = st.text_input("🔗 URL File:", placeholder="https://example.com/data_siswa.csv") 
 
    if file_url: 
        try: 
            # Membaca file berdasarkan ekstensi
            if file_url.endswith(".csv"): 
                df = pd.read_csv(file_url) 
            elif file_url.endswith((".xls", ".xlsx")): 
                df = pd.read_excel(file_url) 
            else: 
                st.error("❌ Format file tidak didukung. Gunakan .csv atau .xlsx") 
                st.stop()
 
            # Menampilkan preview data
            st.subheader("📋 Preview Data:")
            st.dataframe(df.head())
            
            # Validasi kolom yang diperlukan
            required_columns = []
            pelajaran = ['Agama', 'PKN', 'Indo', 'Mate', 'IPA', 'IPS',  
                        'Inggris', 'Senbud', 'PJOK', 'Prakarya', 'B_Daerah']
            
            for pel in pelajaran:
                for sem in range(1, 7):
                    required_columns.append(f"{pel}{sem}")
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Kolom yang hilang: {missing_columns}")
                st.stop()
            
            # Preprocessing dan prediksi
            features_df = df[required_columns]
            df_scaled = scaler.transform(features_df) 
            predictions = model.predict(df_scaled) 
            
            # Karena tidak menggunakan label encoder, langsung gunakan hasil prediksi
            df["Rekomendasi Jurusan"] = predictions
            
            # Tambahkan probabilitas jika tersedia
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df_scaled)
                classes = model.classes_
                
                for i, kelas in enumerate(classes):
                    df[f"Prob_{kelas}"] = probabilities[:, i]
 
            st.success("✅ Prediksi Berhasil!") 
            st.subheader("📊 Hasil Prediksi:")
            st.dataframe(df) 
 
            # Statistik hasil prediksi
            st.subheader("📈 Statistik Hasil:")
            jurusan_counts = df["Rekomendasi Jurusan"].value_counts()
            st.bar_chart(jurusan_counts)
            
            for jurusan, count in jurusan_counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"**{jurusan}**: {count} siswa ({percentage:.1f}%)")
 
            # Download hasil
            csv_data = df.to_csv(index=False)
            st.download_button( 
                "📥 Unduh Hasil Prediksi",  
                data=csv_data,  
                file_name="hasil_prediksi.csv",  
                mime="text/csv" 
            ) 
 
        except Exception as e: 
            st.error(f"⚠️ Gagal membaca atau memproses file. Detail error:\n{e}")

# ---------------------------------------- 
# 🔹 Opsi 3: Upload File Lokal 
# ---------------------------------------- 
st.markdown("---")
st.subheader("📁 Atau Upload File Lokal")

uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # Membaca file berdasarkan tipe
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Preview data
        st.subheader("📋 Preview Data:")
        st.dataframe(df.head())
        
        # Validasi kolom
        required_columns = []
        pelajaran = ['Agama', 'PKN', 'Indo', 'Mate', 'IPA', 'IPS',  
                    'Inggris', 'Senbud', 'PJOK', 'Prakarya', 'B_Daerah']
        
        for pel in pelajaran:
            for sem in range(1, 7):
                required_columns.append(f"{pel}{sem}")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"⚠️ Kolom yang hilang: {missing_columns}")
            st.info("Pastikan file memiliki kolom dengan format: NamaPelajaran + Nomor Semester (contoh: Mate1, Mate2, dst.)")
        else:
            if st.button("🔍 Prediksi untuk Semua Data"):
                # Preprocessing dan prediksi
                features_df = df[required_columns]
                df_scaled = scaler.transform(features_df)
                predictions = model.predict(df_scaled)
                
                # Tambahkan hasil prediksi
                df["Rekomendasi Jurusan"] = predictions
                
                # Tambahkan probabilitas jika tersedia
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(df_scaled)
                    classes = model.classes_
                    
                    for i, kelas in enumerate(classes):
                        df[f"Prob_{kelas}"] = probabilities[:, i]
                
                st.success("✅ Prediksi Berhasil!")
                st.subheader("📊 Hasil Prediksi:")
                st.dataframe(df)
                
                # Statistik
                st.subheader("📈 Statistik Hasil:")
                jurusan_counts = df["Rekomendasi Jurusan"].value_counts()
                st.bar_chart(jurusan_counts)
                
                # Download
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Unduh Hasil Prediksi",
                    data=csv_data,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"❌ Error saat memproses file: {e}")

# Footer
st.markdown("---")
st.markdown("💡 **Tips**: Pastikan file input memiliki kolom nilai untuk setiap mata pelajaran dan semester dengan format yang benar.")
st.markdown("🔧 **Format Kolom**: NamaPelajaran + Nomor Semester (contoh: Mate1, Mate2, IPA1, IPA2, dst.)")