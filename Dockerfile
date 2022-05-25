FROM python:3.8-slim-bullseye

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /app

# CMD streamlit run --server.port 8080 --server.enableCORS false app.py
CMD streamlit run app.py
 
