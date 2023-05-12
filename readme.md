```
docker build -t shelfy .
```
```
docker run -it --rm  -v $(pwd):/usr/src/app/code shelfy /bin/bash 
```
```
poetry run python src/main.py
```