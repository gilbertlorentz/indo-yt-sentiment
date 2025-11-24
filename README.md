### Youtube Sentiment Analysis (Indonesian Content)

=== CREATE DIRECTORY
cd X:\projects\yt-sentiment

=== CREATE ENVIRONTMENT
python -m venv venv

venv\Scripts\activate.bat

pip install -r requirements.txt

=== SETUP YT API KEY
set YT_API_KEY=AIzaSyArfcJV92cTxLuFxPe3nS-wWqK1pZzjEL0

set FLASK_SECRET=some-secret

=== INSTALL TRANSFORMERS
pip install torch transformers

=== RUN TEST.PY
python test.py

=== RUN APP.PY
python app.py
