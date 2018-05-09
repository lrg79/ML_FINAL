# Replicating Tournament Results

1. Clone Facenet from Github Repo (open-source) https://github.com/davidsandberg/facenet
2. Replace `facenet/src/classifier.py` with the supplied `classifier.py` file.
3. Copy `join.py` to `facenet/src/`
4. Download pre-trained model (Casia-WebFace) from GitHub main page https://github.com/davidsandberg/facenet
and store in `facenet/models/`
5. Create a new directory called `Datasets` in `facenet/src/`
6. Copy over training and test image directories into the new `facenet/src/Datasets` folder,
and make sure to put the test image directory into a new directory called `test` (so
you have a nested folder).
7. Copy the `facenet/src/align/align_dataset_mtcnn.py` file to the `src` folder.
8. Perform image alignment for training and test data sets using the following commands:
```bash
python align_dataset_mtcnn.py Datasets/images-train Datasets/images-train-align --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25

python align_dataset_mtcnn.py Datasets/test Datasets/test-align --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
```
9. Perform training using the following comamnds:
```bash
python src/classifier.py TRAIN Datasets/images-train-align models/model-20180408-102900.pb ~/models/kardashians.pkl --batch_size 256
python src/classifier.py CLASSIFY Datasets/test-align models/model-20180408-102900.pb ~/models/kardashians.pkl --batch_size 256
```
10. Finish formatting output by running `python join.py`
11. View results in `newoutput.csv`

