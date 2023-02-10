from pathlib import Path

import pdf2image

PATH_PDF = Path("data/pdf")
PATH_JPG = Path("data/jpg")

for path in PATH_PDF.glob("*.pdf"):
    images = pdf2image.convert_from_path(path)
    for i, image in enumerate(images):
        file_name = Path(PATH_JPG, f"{path.stem + '_' + str(i)}").with_suffix('.jpg')
        image.save(file_name, 'JPEG')

