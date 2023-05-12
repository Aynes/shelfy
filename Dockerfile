FROM python:3

WORKDIR /usr/src/app

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
COPY poetry.lock /usr/src/app/poetry.lock
COPY pyproject.toml /usr/src/app/pyproject.toml
RUN poetry update

RUN apt-get update && apt-get install -y poppler-utils libgl1 tesseract-ocr zbar-tools

ENTRYPOINT /bin/bash