# postgres:
- install pgvector: 
  C:\Program Files\PostgreSQL\18
  use -> vector.v0.8.1-pg18 -> https://github.com/andreiramani/pgvector_pgsql_windows/releases

LINUX: sudo -u postgres psql

set "PGROOT=C:\Program Files\PostgreSQL\18"
cd %TEMP%
git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install

- Enable pgvector extenstion
  psql -U postgres

- Create ragdb
CREATE DATABASE ragdb;
\c ragdb
CREATE EXTENSION IF NOT EXISTS vector;

- Set environment variables
$env:PGHOST="localhost"
$env:PGPORT="5432"
$env:PGUSER="postgres"
$env:PGPASSWORD="postgres"
$env:PGDATABASE="ragdb"
$env:EMBEDDING_TYPE="local"
$env:OPENAI_API_KEY="Your API key"

--using openAI for embeding
# $env:EMBEDDING_TYPE="openai"
$env:OPENAI_API_KEY="Your API key"

- Create environment and install requirements
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


- Run the website using :
  uvicorn app:app --reload

  