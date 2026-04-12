# Запуск контейнера PaddleOCR
`docker run -it --gpus all --name ocr -p 8000:8000 -v ${PWD}:/paddle --shm-size=8g paddleocr_final /bin/bash` \
В итоге надо будет пофиксить, сделать docker обёртку для запуска на любой машине