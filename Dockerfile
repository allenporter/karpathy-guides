FROM python:3.11

RUN pip install --upgrade pip
RUN pip3 install jupyter ipykernel
COPY requirements.txt .
RUN pip3 install -r requirements.txt

