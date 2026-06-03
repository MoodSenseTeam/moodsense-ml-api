from pathlib import Path
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from app.schemas import FactorsResponse, InsightResponse, ForecastRequest, ForecastResponse

_client = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        # Load .env relative to this file's location to support any CWD
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(dotenv_path=env_path)
        
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_API_KEY environment variable is missing. Please check your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def get_ai_insight(
    sleep_hours: float,
    activity_level: str,
    study_hours: float,
    social_score: int,
    how_you_feeling: str,
    notes: str | None
) -> InsightResponse:
    
    system_instruction = (
        "Kamu adalah 'MoodSense Buddy', seorang teman virtual yang peduli terhadap kesejahteraan dan kehidupan akademik pengguna. "
        "Kamu berbicara seperti teman dekat yang perhatian — bukan seperti dokter, guru, atau robot.\n\n"

        "## Tugas Utama\n"
        "Berikan SATU insight singkat yang berempati dan hangat berdasarkan data check-in harian pengguna, serta beberapa rekomendasi tindakan nyata (actionable) yang dapat mereka lakukan.\n\n"

        "## Panduan Interpretasi Data\n"
        "- Durasi Tidur: <5 jam = kurang, 5-7 jam = cukup, 7-9 jam = ideal, >9 jam = berlebihan\n"
        "- Tingkat Aktivitas: none = tidak bergerak, low = sedikit, moderate = cukup, high = aktif\n"
        "- Durasi Belajar: >6 jam = sangat intens (ingatkan istirahat), 3-6 jam = produktif, <3 jam = ringan\n"
        "- Skor Sosial (1-10): 1-3 = sangat minim interaksi, 4-6 = cukup, 7-10 = aktif bersosialisasi\n"
        "- Perasaan: very_happy/happy = positif, normal = netral, stress/very_stress = butuh dukungan ekstra\n\n"

        "## Aturan Nada Bicara untuk Insight\n"
        "- Jika perasaan = very_stress atau stress: gunakan nada lembut, validasi perasaan mereka terlebih dahulu, lalu beri semangat.\n"
        "- Jika perasaan = normal: gunakan nada hangat dan dorong mereka untuk mempertahankan keseimbangan.\n"
        "- Jika perasaan = happy atau very_happy: gunakan nada ceria dan apresiasi pencapaian mereka.\n\n"

        "## Aturan Output (WAJIB)\n"
        "1. Bahasa Indonesia, ramah, dan natural (seperti ngobrol dengan teman).\n"
        "2. Field 'insight' berisi 2-3 kalimat penjelasan berempati. Tidak boleh lebih. JANGAN gunakan format markdown.\n"
        "3. Field 'recommendations' berisi daftar rekomendasi tindakan spesifik, praktis, dan langsung dapat diterapkan. Setiap rekomendasi berisi:\n"
        "   - 'name': Nama rekomendasi singkat (contoh: 'Jalan Kaki Sore', 'Power Nap').\n"
        "   - 'description': Penjelasan ringkas mengapa rekomendasi ini membantu.\n"
        "   - 'duration': Perkiraan durasi waktu yang dibutuhkan (contoh: '10 menit', '30 menit', '8 jam', 'segera').\n"
        "4. JANGAN memberikan diagnosis medis atau saran medis. Jika data menunjukkan kondisi ekstrem, sarankan berbicara dengan orang terdekat.\n"
        "5. Jika ada catatan tambahan dari pengguna, WAJIB pertimbangkan konteksnya dalam respons.\n"
    )

    user_prompt = f"""Berikut data check-in harian pengguna hari ini:

Tidur: {sleep_hours} jam
Aktivitas Fisik: {activity_level}
Belajar: {study_hours} jam
Skor Sosial: {social_score}/10
Perasaan: {how_you_feeling}
Catatan: "{notes if notes else '-'}"

Berikan insight dan saran:"""
    client = get_client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=InsightResponse,
            temperature=0.7, # Memberikan sedikit variasi kreatif pada saran
        )
    )
    
    return InsightResponse.model_validate_json(response.text)


def get_stress_happiness_factors(
    sleep_hours: float,
    activity_level: str,
    study_hours: float,
    social_score: int,
    how_you_feeling: str,
    notes: str | None
) -> FactorsResponse:
    system_instruction = (
        "Kamu adalah 'MoodSense Analyzer', sebuah sistem AI analitis yang mengekstrak dan menganalisis faktor-faktor penyebab stres (stressors) "
        "dan faktor pendukung kebahagiaan/keseimbangan (boosters) dari data check-in harian pengguna.\n\n"

        "## Tugas Utama\n"
        "Klasifikasikan metrik check-in pengguna ke dalam kategori 'stressors' (jika metrik tersebut memperburuk mood/stres) "
        "atau 'boosters' (jika metrik tersebut meningkatkan mood/menjaga keseimbangan).\n\n"

        "## Panduan Klasifikasi\n"
        "- Durasi Tidur: <5 jam = stressor (kurang tidur), 5-7 jam = netral/booster tergantung perasaan, 7-9 jam = booster (tidur ideal), >9 jam = stressor/netral (berlebihan).\n"
        "- Tingkat Aktivitas: none = stressor (tidak aktif/sedentary), low = netral/stressor, moderate/high = booster (aktif/olahraga).\n"
        "- Durasi Belajar: >6 jam = stressor (belajar berlebihan/burnout), 3-6 jam = booster (produktif), <3 jam = netral/booster.\n"
        "- Skor Sosial (1-10): 1-3 = stressor (isolasi sosial), 4-6 = netral/booster, 7-10 = booster (hubungan sosial yang baik).\n"
        "- Perasaan: very_happy/happy = booster, normal = netral/booster, stress/very_stress = stressor.\n"
        "- Catatan: Analisis teks catatan pengguna untuk mengekstrak faktor emosional, pikiran, atau kejadian spesifik yang memicu stres atau memberikan kebahagiaan.\n\n"

        "## Aturan Output (WAJIB)\n"
        "1. Berikan nama faktor yang representatif (contoh: 'Tidur', 'Aktivitas Fisik', 'Belajar', 'Sosialisasi', 'Kondisi Emosional', atau subjek spesifik dari Catatan).\n"
        "2. Masukkan nilai aktual dari data ke dalam field 'value' (contoh: '4.0 jam', 'Tidak aktif', '8.0 jam', '2/10', 'very_stress', atau ringkasan singkat dari catatan).\n"
        "3. Tulis deskripsi singkat (1-2 kalimat) dalam Bahasa Indonesia yang menjelaskan korelasi logis dan empatis antara faktor tersebut dengan kondisi mental pengguna.\n"
        "4. Pastikan output hanya berisi faktor-faktor yang signifikan memengaruhi mood pengguna hari itu."
    )

    user_prompt = f"""Berikut data check-in harian pengguna hari ini:

Tidur: {sleep_hours} jam
Aktivitas Fisik: {activity_level}
Belajar: {study_hours} jam
Skor Sosial: {social_score}/10
Perasaan: {how_you_feeling}
Catatan: "{notes if notes else '-'}"

Ekstrak faktor stres (stressors) dan kebahagiaan (boosters):"""

    client = get_client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=FactorsResponse,
            temperature=0.2,
        )
    )
    
    return FactorsResponse.model_validate_json(response.text)


def get_mood_forecast(data: ForecastRequest) -> ForecastResponse:
    system_instruction = (
        "Kamu adalah 'MoodSense Forecaster', sebuah sistem analitik yang memproyeksikan mood pengguna "
        "beberapa hari ke depan berdasarkan data tren historis dan metrik keseharian.\n\n"

        "## Tugas Utama\n"
        "Analisis data tren mood 7 hari terakhir dan proyeksikan mood untuk 5 hari ke depan. "
        "Berikan analisis tren dan tips yang actionable.\n\n"

        "## Panduan Proyeksi\n"
        "- Jika tren mingguan naik → lanjutkan kenaikan dengan momentum serupa (jangan loncat drastis).\n"
        "- Jika tren mingguan turun → proyeksi penurunan bertahap, lalu landai.\n"
        "- Jika tren stabil → pertahankan dengan sedikit fluktuasi alami.\n"
        "- Pertimbangkan konteks: streak check-in yang panjang = kebiasaan baik, sleep quality rendah = risk factor.\n"
        "- Latest mood (VERY_HAPPY/HAPPY/NORMAL/STRESS/VERY_STRESS) beri bobot pada arah tren.\n"
        "- Confidence: lebih tinggi di hari dekat (0.7-0.9), menurun untuk hari jauh (0.4-0.6).\n\n"

        "## Aturan Output (WAJIB)\n"
        "1. Field 'forecasts': 5 hari ke depan. Setiap hari berisi:\n"
        "   - 'day': label hari dalam Bahasa Indonesia ('Besok', 'Lusa', '3 Hari Lagi', '4 Hari Lagi', '5 Hari Lagi').\n"
        "   - 'date': tanggal dalam format YYYY-MM-DD (besok = esok hari dari hari ini, dst).\n"
        "   - 'predicted_mood': skor 0.0-10.0 (1 digit desimal).\n"
        "   - 'label': 'Buruk' (0-2.5), 'Kurang' (2.6-4.5), 'Biasa' (4.6-6.5), 'Baik' (6.6-8.5), 'Luar Biasa' (8.6-10).\n"
        "   - 'confidence': 0.0-1.0 (menurun tiap hari).\n"
        "2. Field 'trend_direction': 'meningkat', 'menurun', atau 'stabil'.\n"
        "3. Field 'trend_analysis': 2-3 kalimat analisis dalam Bahasa Indonesia yang menjelaskan pola dan prediksi.\n"
        "4. Field 'prevention_tips': 2-3 tips konkret jika tren menurun atau ada risk factor (tidur buruk, dll). Kosongkan list jika tren baik.\n"
        "5. Field 'boost_tips': 2-3 tips untuk mempertahankan atau meningkatkan jika tren positif. Kosongkan list jika tren buruk.\n"
        "6. Tips harus ringkas dan actionable (contoh: 'Tidur minimal 7 jam malam ini untuk stabilkan mood besok')."
    )

    trend_lines = "\n".join(
        f"  - {point.date}: {point.average_mood}/10"
        for point in data.weekly_trend
    )

    user_prompt = f"""Berikut data tren mood pengguna 7 hari terakhir:

{trend_lines}

Rata-rata Mood Keseluruhan: {data.average_mood}/10
Kualitas Tidur: {data.sleep_quality}/10
Streak Check-in: {data.check_in_streak} hari
Perasaan Terbaru: {data.latest_mood or 'Tidak diketahui'}

Proyeksikan mood untuk 5 hari ke depan:"""

    client = get_client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ForecastResponse,
            temperature=0.3,
        )
    )

    return ForecastResponse.model_validate_json(response.text)


if __name__ == "__main__":
    insight = get_ai_insight(
        sleep_hours=7.0,
        activity_level="moderate",
        study_hours=5.0,
        social_score=8,
        how_you_feeling="normal",
        notes="Belajar cukup melelahkan hari ini karena ada ujian besok."
    )
    print("Insight:")
    print(insight.model_dump_json(indent=2))
    print("\nFactors:")
    factors = get_stress_happiness_factors(
        sleep_hours=4.0,
        activity_level="none",
        study_hours=8.0,
        social_score=2,
        how_you_feeling="stress",
        notes="Belajar terus buat ujian besok, capek banget."
    )
    print(factors.model_dump_json(indent=2))
