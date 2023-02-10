```
docker build -t shelfy .
```
```
docker run -it --rm  -v $(pwd):/usr/src/app shelfy /bin/bash 
```
```
export PATH="/root/.local/bin:$PATH"
```
```
poetry update
```
```
poetry run python src/pdf_to_image.py 
```