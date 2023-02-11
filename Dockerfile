FROM python:3

WORKDIR /usr/src/app
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN export PATH="/root/.local/bin:$PATH"
RUN apt-get update
RUN apt-get install -y poppler-utils
RUN apt-get update && apt-get install -y libgl1
RUN apt-get install -y tesseract-ocr
ENTRYPOINT /bin/bash