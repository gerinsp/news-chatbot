import requests
import csv

questions = [
    "Apa kegiatan Bupati Sragen pada bulan Mei?",
    "Bagaimana kondisi pasar hewan qurban di Banjarnegara?",
    "Apa manfaat sektor pariwisata menurut Siti Mukaromah?",
    "Apa saja yang dilakukan dalam pelayanan gratis saat Waisak di Borobudur?",
    "Apa tema perayaan Waisak tahun ini dan pesannya bagi umat Buddha?",
    "Siapa saja yang dilibatkan PKK Kendal dalam transformasi digital?",
    "Apa kontribusi Pasar Medang terhadap UMKM lokal?",
    "Bagaimana kisah Fathkurohman dalam menunaikan ibadah haji?",
    "Apa penyebab kebakaran di TPA Jatibarang Semarang?",
    "Mengapa warga Pati resah dengan tanggul Sungai Widodaren?",
    "Bagaimana tradisi Mragat Kerbau di Grobogan dijalankan?",
    "Mengapa lulusan SMK di Blora mendominasi pencari kartu kerja?",
    "Apa perkembangan terbaru kasus PPDS Dokter Aulia di Semarang?",
    "Bagaimana sikap Jateng terkait siswa bermasalah dan barak militer?",
    "Apa makna dari tradisi Pisowanan Balasan di Demak?",
    "Bagaimana cara jamaah haji Blora menghindari koper tertukar?",
    "Mengapa tiga jamaah haji dari Pati gagal berangkat?",
    "Apa tuntutan mahasiswa UKSW dalam demonstrasi mereka?",
    "Apa penyebab banjir di Sukoharjo menurut warga?",
    "Apa dampak perbaikan Jalan Gentan–Bekonang bagi warga?",
    "Apa penyebab banjir di Grobogan pada Mei 2025?"
]

API_URL = "http://localhost:8001/chat/"

output_csv = "jawaban_chatbot.csv"

results = []

for idx, q in enumerate(questions, start=1):
    try:
        print(f"Mengirim pertanyaan #{idx}: {q}")
        payload = {
            "user_input": q,
            "history": []
        }
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        answer = response.json().get("answer", "(tidak ada jawaban)")
    except Exception as e:
        answer = f"ERROR: {str(e)}"
    results.append([idx, q, answer])

with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["No", "Pertanyaan", "Jawaban"])
    writer.writerows(results)

print(f"\nSemua hasil disimpan ke {output_csv}")
