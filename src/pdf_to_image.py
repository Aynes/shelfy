from pathlib import Path

import pdf2image

def pdf_to_jpg(path_to_pdf, path_to_jpg):
    for path in path_to_pdf.glob("*.pdf"):
        images = pdf2image.convert_from_path(path)
        for i, image in enumerate(images):
            file_name = Path(path_to_jpg, f"{path.stem + '_' + str(i)}").with_suffix('.jpg')
            image.save(file_name, 'JPEG')

