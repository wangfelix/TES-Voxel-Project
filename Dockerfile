FROM ubuntu:20.04

RUN apt update
RUN apt install python3 -y

WORKDIR /src

COPY requirements.txt
RUN pip install -r requirements.txt

COPY VAE_Test.py .
CMD ["python", "VAE_Test.py"]