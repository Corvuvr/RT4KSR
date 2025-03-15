# Towards Real-Time 4K Image Super-Resolution

Официальный README можно найти [здесь](https://github.com/eduardzamfir/RT4KSR/blob/main/README.md).

## Установка

- Создать conda-окружение:
```
conda create --name RTSR python==3.10
source activate RTSR
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia
pip install -r requirements.txt
```
---
## Использование

Укажите директорию HR-изображений с помощью аргумента `--dataroot`: модель бикубически масштабирует их до LR-разрешения и применяет инференс. Результаты сохраняются в `results/<dataroot-parent-folder>`.
Все детали можно найти в `infer.py`. Аргумент `--is-train` всегда необходим, поскольку для репараметризации требуется предварительно загруженная обучающая архитектура.
Укажите суффикс выходных изображений, если необходимо, с помощью аргумента `--suffix`.
````
python code/infer.py --checkpoint-id rt4ksr_[x2|x3] --scale [2|3] --arch rt4ksr_rep --is-train --dataroot [DATAROOT] --suffix [SUFFIX]
````
Примеры:
```
python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/DIV2K/DIV2K/ --suffix "x4_out"
python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/General100/General100   
python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/BSDS100/BSDS100         
python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/urban100/urban100       
python code/infer.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train --dataroot ../datasets/set14/set14/set14       
```
---