# Gunakan image dasar Python 3.10
FROM python:3.10-slim

# Setel direktori kerja dalam kontainer
WORKDIR /app

# Salin file requirements.txt ke dalam kontainer
COPY requirements.txt requirements.txt

# Instal dependensi dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file dari direktori lokal ke dalam kontainer
COPY . .

# Eksekusi aplikasi menggunakan gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
