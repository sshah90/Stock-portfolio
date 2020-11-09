FROM python:3.7.4-slim-stretch

COPY requirement.txt requirement.txt

RUN pip install -r requirement.txt

# copy code base in docker image
COPY . .

# exposing default port for streamlit
EXPOSE 8501

# run app
CMD streamlit run frontend.py
