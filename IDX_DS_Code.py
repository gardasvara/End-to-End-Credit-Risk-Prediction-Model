# %% [markdown]
# # End-to-End Credit Risk Prediction Model
# ### ID/X - Partner - Data Science Februari 2026
# ### Gardasvara Mistortoify

# %% [markdown]
# ## 1. IMPORT LIBRARIES & LOAD DATA
# File loan_data_2007_2014.csv sudah ada di folder direktori yang sama

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# %%
df = pd.read_csv('loan_data_2007_2014.csv', low_memory=False)
print("Data berhasil di-load!")

# %% [markdown]
# ## 2. EDA & DEFINING TARGET VARIABLE (LABELING)
# Fokus memprediksi 'Bad Loan'. 
# Bad Loan = 1, Good Loan = 0

# %%
def define_target(status):
    if status in ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 
                  'Does not meet the credit policy. Status:Default']:
        return 1
    else:
        return 0

df['bad_loan'] = df['loan_status'].apply(define_target)

# Cek proporsi target (Imbalance Check)
print(df['bad_loan'].value_counts(normalize=True))

# %% [markdown]
# ### 1. Visualisasi Target Variable

# %%
sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
ax = sns.countplot(x='bad_loan', data=df, palette='viridis')
plt.title('Distribusi Good Loan (0) vs Bad Loan (1)')
plt.xlabel('Status Pinjaman (0=Good, 1=Bad)')
plt.ylabel('Jumlah Nasabah')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + 0.3, p.get_height() + 100))
plt.show()

# %% [markdown]
# ### 2. Korelasi Fitur Numerik dengan Target
# Ambil beberapa kolom kunci berdasarkan Data Dictionary

# %%
cols_eda_num = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
plt.figure(figsize=(8, 6))
sns.heatmap(df[cols_eda_num + ['bad_loan']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasi Fitur Numerik Utama')
plt.show()



# %% [markdown]
# 1. int_rate (Suku Bunga) - Korelasi Positif Kuat
# 
#     Semakin tinggi suku bunga, semakin tinggi risiko gagal bayar. Ini wajar karena peminjam berisiko tinggi memang dikenakan bunga lebih besar sejak awal.
# 
# 2. dti (Debt-to-Income Ratio) - Korelasi Positif
# 
#     Semakin tinggi rasio utang terhadap pendapatan, semakin besar kemungkinan default. Peminjam dengan beban utang yang sudah menumpuk lebih rentan gagal bayar.
# 
# 3. revol_util (Revolving Line Utilization) - Korelasi Positif
# 
#     Penggunaan limit kredit yang mendekati maksimal (misal kartu kredit mentok) mengindikasikan kesulitan likuiditas nasabah.
# 
# 4. annual_inc (Pendapatan Tahunan) - Korelasi Negatif (Biasanya Lemah)
# 
#     Semakin tinggi pendapatan, risiko gagal bayar cenderung turun, namun seringkali korelasinya tidak sekuat int_rate.

# %% [markdown]
# ### 3. Analisis Bivariate: Grade vs Bad Loan

# %%
plt.figure(figsize=(10, 5))
sorted_grades = sorted(df['grade'].unique())
sns.barplot(x='grade', y='bad_loan', data=df, order=sorted_grades, palette='coolwarm', errorbar=None)
plt.title('Persentase Gagal Bayar (Bad Loan) berdasarkan Grade')
plt.ylabel('Probabilitas Bad Loan')
plt.xlabel('Grade Pinjaman')
plt.show()

# %% [markdown]
# Insight: Grafik naik dari A ke G, berarti Grade berfungsi dengan baik (Riskier loans have higher grades).

# %% [markdown]
# ### 4. Analisis Bivariate: Term (Tenor) vs Bad Loan

# %%
plt.figure(figsize=(6, 4))
sns.barplot(x='term', y='bad_loan', data=df, palette='muted', errorbar=None)
plt.title('Persentase Gagal Bayar berdasarkan Tenor (Term)')
plt.ylabel('Probabilitas Bad Loan')
plt.show()

# %% [markdown]
# Insight: Tenor panjang (60 bulan) memiliki risiko lebih tinggi daripada 36 bulan.

# %% [markdown]
# ## 3. DATA CLEANING & FEATURE SELECTION
# 1. Membuang Sampah (Drop Useless): Menghapus kolom seperti ID, URL, dan Deskripsi yang tidak memiliki pola statistik untuk prediksi.
# 
# 2. Mencegah "Kecurangan" (Drop Leakage): Menghapus kolom "masa depan" (seperti total pembayaran yang sudah diterima). Jika kolom ini ada, model akan tahu jawabannya (apakah lunas/gagal) sebelum memprediksi, yang membuat model tidak valid dipakai untuk peminjam baru.
# 
# 3. Efisiensi Data (Drop High Missing Value): Kolom yang isinya lebih dari 50% kosong dibuang karena terlalu sedikit informasinya untuk dipelajari.
# 
# 4. Konversi Teks ke Angka:
# 
#     - term: Dari "36 months" menjadi angka 36.0.
# 
#     - emp_length: Dari "10+ years" menjadi angka 10.0. Ini penting agar mesin bisa menghitungnya secara matematis.

# %%
# A. Drop Kolom Unik/Tidak Berguna
cols_to_drop = ['id', 'member_id', 'url', 'desc', 'policy_code', 'zip_code', 'addr_state', 'title', 'emp_title']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# B. Drop Kolom Data Leakage
leakage_cols = ['recoveries', 'collection_recovery_fee', 'total_rec_prncp', 'total_rec_int', 
                'total_rec_late_fee', 'total_pymnt', 'total_pymnt_inv', 'last_pymnt_d', 
                'last_pymnt_amnt', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv']
df.drop(columns=leakage_cols, inplace=True, errors='ignore')

# C. Drop Kolom dengan Missing Value > 50%
threshold = len(df) * 0.5
df.dropna(thresh=threshold, axis=1, inplace=True)

# D. Feature Engineering Sederhana
# Membersihkan 'term' (contoh: " 36 months" -> 36)
df['term'] = df['term'].astype(str).str.replace(' months', '').astype(float)

# Membersihkan 'emp_length' (contoh: "10+ years" -> 10)
df['emp_length'] = df['emp_length'].astype(str).str.replace(r'\D', '', regex=True)
df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce') # Ubah ke angka, yang error jadi NaN
df['emp_length'] = df['emp_length'].fillna(0) # Fill NA dengan 0 tanpa inplace=True

print("Step 3 Selesai: Data Cleaning Berhasil.")

# %% [markdown]
# #### VERIFIKASI HASIL DATA CLEANING & FEATURE SELECTION

# %%
print("1. Cek Ukuran Data (Baris, Kolom):")
print(df.shape)

print("\n2. Cek Tipe Data 'term' dan 'emp_length':")
print(df[['term', 'emp_length']].dtypes)

print("\n3. Cek Sampel Data 'term' dan 'emp_length':")
print(df[['term', 'emp_length']].head())

print("\n4. Cek Apakah Kolom Leakage Masih Ada?")
leakage_check = 'recoveries' in df.columns
print(f"Kolom 'recoveries' ada di dataframe? {leakage_check}")

# %% [markdown]
# ## 4. PREROCESSING & FEATURE ENGINEERING 
# 1. Menambal Data Bolong (Imputation):
# 
#     - Model Machine Learning (seperti Logistic Regression) matematika-nya akan error jika bertemu nilai kosong (NaN).
# 
#     - Solusi: Isi kekosongan tersebut. Untuk kolom angka, isi dengan nilai tengah (median). Untuk kolom kategori/teks, isi dengan data yang paling sering muncul (mode).
# 
# 2. Membuang Fitur yang Belum Perlu :
# 
#     - Membuang kolom tanggal (issue_d, earliest_cr_line) karena model tidak bisa langsung membaca format tanggal "Jan-2010".
# 
# 3. One-Hot Encoding :
# 
#     - Model tidak mengerti teks seperti "RENT", "OWN", atau "MORTGAGE".
# 
#     - Solusi: Ubah menjadi angka "1" dan "0".
# 
#     - Contoh: Kolom home_ownership akan pecah menjadi home_ownership_RENT, home_ownership_OWN, dll. Jika nasabah menyewa rumah, maka home_ownership_RENT bernilai 1, sisanya 0.

# %%
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns


for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

# Membuang 'loan_status' karena sudah ada 'bad_loan'
# Kita juga membuang 'issue_d' dkk karena format tanggal sulit diproses model sederhana
features = df.drop(['bad_loan', 'loan_status', 'issue_d', 'earliest_cr_line', 'last_credit_pull_d'], axis=1, errors='ignore')
target = df['bad_loan']

# One-Hot Encoding
features = pd.get_dummies(features, drop_first=True)

print("Step 4 Selesai: Preprocessing Berhasil.")

# %% [markdown]
# ### VERIFIKASI HASIL STEP 4

# %%
print("1. Cek Sisa Missing Values:")
total_nan = features.isnull().sum().sum()
print(f"Jumlah nilai kosong di dataset: {total_nan}")

print("\n2. Cek Tipe Data (Harus Angka Semua):")
non_numeric = features.select_dtypes(include=['object']).columns
print(f"Jumlah kolom bertipe object/teks: {len(non_numeric)}")

print("\n3. Cek Jumlah Kolom Setelah Encoding:")
print(f"Shape awal (sebelum encoding): {df.shape}")
print(f"Shape akhir (features siap training): {features.shape}")

print("\n4. Data Siap Pakai (5 Baris Pertama):")
print(features.head())

# %% [markdown]
# ## 5. MODELING (GRADIENT BOOSTING)
# Tahap ini menggunakan algoritma Gradient Boosting Classifier yang bekerja dengan cara membangun sekumpulan pohon keputusan secara bertahap, di mana setiap pohon baru bertugas mengoreksi kesalahan prediksi dari pohon sebelumnya. Model ini diinisialisasi dengan parameter seperti n_estimators untuk menentukan jumlah pohon dan learning_rate untuk mengatur kekuatan koreksi setiap tahap guna mencapai akurasi maksimal. Proses training dilakukan pada data yang telah melalui standardisasi agar model dapat secara optimal mempelajari pola risiko kredit dari fitur-fitur yang tersedia dalam dataset.

# %%
from sklearn.ensemble import GradientBoostingClassifier

# 1. Split Data 
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    target, 
    test_size=0.2, 
    random_state=42, 
    stratify=target
)

# 2. Scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Training Model Gradient Boosting
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    validation_fraction=0.1, 
    n_iter_no_change=10      
)

print("Training Gradient Boosting Model...")
model.fit(X_train_scaled, y_train)

# 4. Prediksi
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Step 5 Selesai: Model Gradient Boosting berhasil dilatih.")

# %% [markdown]
# ## 6. EVALUATION MODEL
# 1. Recall (untuk kelas 1/Bad Loan): Seberapa banyak Bad Loan yang berhasil ditebak benar oleh model.
# 
# 2. ROC-AUC Score: Mengukur kemampuan model membedakan antara Good Loan dan Bad Loan. Semakin dekat ke 1, semakin bagus.
# 
# 3. Confusion Matrix: Tabel untuk melihat detail berapa yang tebakannya benar vs salah.
# 
# 4. Feature Importance (Bar Chart): Untuk mendapatkan insight variabel mana yang paling mempengaruhi

# %%
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Classification Report
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 2. Hitung Skor ROC-AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {auc_score:.4f}")

# 3. Visualisasi Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (AUC = {auc_score:.2f})')
plt.xlabel('Prediksi Model (0=Good, 1=Bad)')
plt.ylabel('Kenyataan (0=Good, 1=Bad)')
plt.show()

# 4. Visualisasi ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'Model (AUC = {auc_score:.2f})', color='orange')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 5. Feature Importance
importances = model.feature_importances_

feat_imp = pd.DataFrame({
    'Feature': features.columns,
    'Importance': importances
})

top_features = feat_imp.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title('Top 10 Fitur Paling Penting (Gradient Boosting)')
plt.xlabel('Tingkat Kepentingan (Importance Score)')
plt.show()


