# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import shap
from fpdf import FPDF
from datetime import datetime
from sklearn.model_selection import train_test_split

# ============================================================
#  FIX: INITIALIZATION (agar tidak KeyError lagi)
# ============================================================
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None

if "df_hasil" not in st.session_state:
    st.session_state["df_hasil"] = None

# ============================================================
#  CSS tampilan (tetap sama)
# ============================================================
st.markdown("""
<style>
/* Background utama */
.stApp {
    background-color: #ffffff;
    color: #000000;
}

/* Font header h2 */
.st-emotion-cache-3uj0rx h2 {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
}

/* Semua teks default jadi hitam */
html, body, [class*="css"] {
    color: #000000 !important;
    font-family: "Arial", sans-serif;
}

/* Input text jadi putih */
input[type="text"], textarea, .stTextInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px;
    padding: 8px;
}

/* Button primary */
div.stButton > button[kind="primary"] {
    background-color: #22C55E;
    color: white !important;
    border: none;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.6em 1em;
    transition: 0.3s;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #16A34A;
}

/* Button secondary / PDF */
div.stDownloadButton > button {
    background-color: #DC3545;
    color: white !important;
    border: none;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.6em 1em;
    transition: 0.3s;
}
div.stDownloadButton > button:hover {
    background-color: #C82333;
}

/* Bar chart container */
.bar-container {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
}
.bar-name {
    width: 25%;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bar-value-container {
    width: 75%;
    background: #eee;
    border-radius: 5px;
    position: relative;
}
.bar-value {
    color: white;
    padding: 2px 6px;
    border-radius: 5px;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 30px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
#  KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Abnormal Parameter Analysis",
    layout="wide"
)

# ============================================================
#  KELAS PDF
# ============================================================
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Laporan Hasil Analisis Anomali', 0, 1, 'C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 10, f"Tanggal Laporan: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

    def vehicle_info_section(self, info):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, '1. Informasi Kendaraan', 0, 1, 'L')
        self.set_font('Helvetica', '', 11)
        for key, value in info.items():
            self.cell(0, 8, f"{key}: {value if value else '-'}", 0, 1)
        self.ln(10)

    def analysis_results_section(self, df_results):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, '2. Hasil Analisis Parameter', 0, 1, 'L')
        
        self.set_font('Helvetica', 'B', 10)
        self.cell(95, 10, 'Parameter', 1, 0, 'C')
        self.cell(35, 10, 'Skor', 1, 0, 'C')
        self.cell(60, 10, 'Prioritas', 1, 1, 'C')

        self.set_font('Helvetica', '', 10)
        for _, row in df_results.iterrows():
            score = row['importance']
            if score > 0.6:
                pr = "Kritis"; rgb = (239, 68, 68)
            elif score > 0.4:
                pr = "Penting"; rgb = (245, 158, 11)
            else:
                pr = "Opsional"; rgb = (34, 197, 94)

            self.cell(95, 8, f" {row['parameter']}", 1, 0, 'L')
            self.cell(35, 8, f"{score:.3f}", 1, 0, 'C')

            self.set_text_color(255, 255, 255)
            self.set_fill_color(*rgb)
            self.cell(60, 8, pr, 1, 1, 'C', True)
            self.set_text_color(0, 0, 0)

@st.cache_data
def create_pdf_report(vehicle_info, df_results):
    pdf = PDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.vehicle_info_section(vehicle_info)
    pdf.analysis_results_section(df_results)
    return pdf.output(dest='S').encode('latin-1')

# ============================================================
#  FUNGSI UTAMA ANALISIS
# ============================================================
@st.cache_data
def run_analysis(df_normal, df_damage):
    df_normal = df_normal.copy()
    df_damage = df_damage.copy()
    df_normal['Label'] = 'Normal'
    df_damage['Label'] = 'Anomali'
    df = pd.concat([df_normal, df_damage], ignore_index=True)

    # Drop fitur
    fitur_dibuang = [
        "SampleTime", "EngineRunTime","EngineSpeed", "VehicleSpeed",
        "IntakeAir", "InitialEngineCoolantTemp", "InitialIntakeAirTemp",
        "AtmospherePressure","ThrottleFullyCloseLearn","ThrottleSensOpenPos#1",
        "KnockFeedbackValue","KnockCorrectLearnValue",
        "FCTAU", "TCandTE1", "ElectricalLoadSignal", "A/CSignal", "StopLightSwitch",
        "PowerSteeringSignal", "StarterSignal", "VVTOCVDuty#1", "VVTChangeAngle#1",
        "VVTAimAngle#1", "EVAPPurgeVSV","OpenSideMalfunction","ST1","ACTVSV",
        "PowerSteer.Sig.Record","FuelSystemStatus#1","FuelPump/SpeedStatus","ActuatorPowerSupply",
        "ETCSActuatorPower","ThrottleMotor","ThrottleSensOpenPos#2","SystemGuard",
        "AcceleratorIdlePosition","ThrottleIdlePosition","VVTControlStatus#1",
        "ClosedThrottlePositionSW","IdleFuelCut","PurgeDensityLearnValue",
        "FailSafeDrive","FailSafeDrive(MainCPU)","CheckMode",
        "SPDTestResult","#Codes(IncludeHistory)","MIL","TimeafterDTCCleared",
        "DistancefromDTCCleared","WarmupCycleClearedDTC","OBDRequirements",
        "NumberofEmissionDTC","ModelCode","EngineType","CylinderNumber",
        "TransmissionType","Destination","ModelYear","SystemIdentification"
    ]
    fitur_dibuang = [f for f in fitur_dibuang if f in df.columns]
    df_reduced = df.drop(columns=fitur_dibuang)

    X = df_reduced.drop(columns=['Label'])
    y = df_reduced['Label'].apply(lambda x: 1 if x == 'Anomali' else 0)

    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X.columns)

    X_normal = X_scaled[y == 0].reset_index(drop=True)
    X_anom = X_scaled[y == 1].reset_index(drop=True)

    X_train_norm, X_test_norm = train_test_split(X_normal, test_size=0.2, random_state=42)

    X_test = pd.concat([X_test_norm, X_anom], ignore_index=True)
    y_test = pd.Series([0]*len(X_test_norm) + [1]*len(X_anom))

    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    model.fit(X_train_norm)

    y_pred_raw = model.predict(X_test)
    y_pred = [1 if v == -1 else 0 for v in y_pred_raw]
    anomaly_score = model.decision_function(X_test)

    X_test_original = pd.DataFrame(scaler.inverse_transform(X_test), columns=X.columns)

    df_hasil = X_test_original.copy()
    df_hasil['True_Label'] = y_test.replace({0:'Normal',1:'Anomali'}).values
    df_hasil['Prediksi'] = y_pred
    df_hasil['Prediksi_Label'] = pd.Series(y_pred).replace({0:'Normal',1:'Anomali'}).values
    df_hasil['Anomaly_Score'] = anomaly_score

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    anomaly_idx = df_hasil.index[df_hasil['Prediksi_Label'] == 'Anomali']

    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[0]

    shap_anom = shap_arr[anomaly_idx]

    if shap_anom.size == 0:
        mean_abs_shap = np.zeros(X.shape[1])
    else:
        mean_abs_shap = np.abs(shap_anom).mean(axis=0)

    if mean_abs_shap.max() > 0:
        importance_norm = (mean_abs_shap - mean_abs_shap.min()) / (mean_abs_shap.max() - mean_abs_shap.min())
    else:
        importance_norm = np.zeros_like(mean_abs_shap)

    df_importance = pd.DataFrame({
        "parameter": X.columns,
        "importance": importance_norm
    }).sort_values("importance", ascending=False)

    return df_importance, df_hasil

# ============================================================
#  UI STREAMLIT
# ============================================================
st.header("Langkah 1: Informasi Kendaraan")
col1, col2 = st.columns(2)
with col1:
    no_polisi = st.text_input("Nomor Polisi")
    no_rangka = st.text_input("Nomor Rangka")
with col2:
    model_kendaraan = st.text_input("Model Kendaraan")
    nama_teknisi = st.text_input("Nama Teknisi")

st.header("Langkah 2: Upload Data")
col3, col4 = st.columns(2)
with col3:
    normal_file = st.file_uploader("Data Normal", type="csv")
with col4:
    damage_file = st.file_uploader("Data Kerusakan", type="csv")

st.markdown("---")

if st.button("Jalankan Analisis", type="primary", use_container_width=True):
    if not normal_file or not damage_file:
        st.error("Upload kedua file terlebih dahulu.")
        st.stop()
    try:
        df_normal = pd.read_csv(normal_file)
        df_damage = pd.read_csv(damage_file)
        df_importance, df_hasil = run_analysis(df_normal, df_damage)

        st.session_state["analysis_results"] = df_importance
        st.session_state["df_hasil"] = df_hasil
        st.session_state["analysis_done"] = True

        st.success("Analisis selesai!")
    except Exception as e:
        st.session_state["analysis_done"] = False
        st.error(f"Error: {e}")

# ============================================================
#  TAMPILKAN HASIL
# ============================================================
if st.session_state.get("analysis_done"):
    results_df = st.session_state.get("analysis_results")
    df_hasil = st.session_state.get("df_hasil")

    if results_df is None or df_hasil is None:
        st.warning("Analisis belum dijalankan.")
        st.stop()

    st.header("Hasil Analisis & Rekomendasi Pemeriksaan")

    colA, colB = st.columns([2.5, 1])
    with colB:
        vehicle_info = {
            "Nomor Polisi": no_polisi,
            "Nomor Rangka": no_rangka,
            "Model Kendaraan": model_kendaraan,
            "Nama Teknisi": nama_teknisi
        }
        pdf_data = create_pdf_report(vehicle_info, results_df)
        st.download_button(
            "📥 Cetak PDF",
            data=pdf_data,
            file_name="Laporan_Analisis.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # === Bar chart responsive & tidak terbungkus ===
    for _, row in results_df.iterrows():
        score = row['importance']
        param = row['parameter']

        if score > 0.6:
            color = "#EF4444"
        elif score > 0.4:
            color = "#F59E0B"
        else:
            color = "#22C55E"

        st.markdown(f"""
        <div class='bar-container'>
            <div class='bar-name' title='{param}'>{param}</div>
            <div class='bar-value-container'>
                <div class='bar-value' style='width:{max(score*100, 10)}%;background:{color}'>{score:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("Lihat tabel hasil prediksi (nilai asli)"):
        st.dataframe(df_hasil, use_container_width=True)
