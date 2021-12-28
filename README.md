# NTO-OCR

### Task
You have to develop an algorithm that can recognize the text in the photo. The model prediction is a text string that matches the text in the picture.


### Data
We were given ~66.000 images, each has handwritten Russian words, which were written by the pupils.

[![image.png](https://i.postimg.cc/0ykSYq0L/image.png)](https://postimg.cc/WhytVCKn)

Also, we added another Dataset from Kaggle, which has ~33.000 images. So, then, we had ~100.000 images for training and validation of our models.

Datasets' links:
- Original Dataset: https://www.kaggle.com/vad13irt/nto-ocr
- Added Dataset: https://www.kaggle.com/constantinwerner/cyrillic-handwriting-dataset 

### Evaluation
The **CER (Character Error Rate)** metric was used as the main metric for evaluating participants' submissions. <br>

[![Screenshot-23.png](https://i.postimg.cc/VL7H8Vkm/Screenshot-23.png)](https://postimg.cc/dLy9RnCx)
<br>
<img src="https://render.githubusercontent.com/render/math?math=dist_c"> - Levenshtein Distance.<br>
<img src="https://render.githubusercontent.com/render/math?math=len_c"> - length of i-th sample.

The **Accuracy** metric is used as an additional metric to evaluate the submissions of the participants.

[![Screenshot-24.png](https://i.postimg.cc/SQGzJ08F/Screenshot-24.png)](https://postimg.cc/14fzjjJJ)

<img src="https://render.githubusercontent.com/render/math?math=pred_i"> - prediction of i-th sample.<br>
<img src="https://render.githubusercontent.com/render/math?math=true_i"> - target  of i-th sample.<br>
<img src="https://render.githubusercontent.com/render/math?math=n"> - amount of samples.<br>


### Model

[![Blank-board.png](https://i.postimg.cc/sgcsFkS9/Blank-board.png)](https://postimg.cc/4YmrVFWm)

### Requirements
* numpy==1.19.5
* torch==1.8.1
* timm==0.4.12
* opencv_python_headless==4.5.2.52
* editdistance==0.6.0
