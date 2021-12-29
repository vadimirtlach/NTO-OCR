# NTO-OCR

### Task
You have to develop an algorithm that can recognize the text in the photo. Model prediction is a text string that matches the text in the picture.

### Evaluation
The **CER (Character Error Rate)** metric is used as the main metric for evaluating participants' submissions. <br>

[![Screenshot-23.png](https://i.postimg.cc/VL7H8Vkm/Screenshot-23.png)](https://postimg.cc/dLy9RnCx)
<br>
<img src="https://render.githubusercontent.com/render/math?math=dist_c"> - Levenshtein distance<br>
<img src="https://render.githubusercontent.com/render/math?math=len_c"> - length of i-th sample

The **Accuracy** metric is used as an additional metric to evaluate the submissions of the participants.

[![Screenshot-24.png](https://i.postimg.cc/SQGzJ08F/Screenshot-24.png)](https://postimg.cc/14fzjjJJ)

<img src="https://render.githubusercontent.com/render/math?math=pred_i"> - prediction of i-th sample<br>
<img src="https://render.githubusercontent.com/render/math?math=true_i"> - target  of i-th sample<br>
<img src="https://render.githubusercontent.com/render/math?math=n"> - amount of samples<br>

### Requirements
* matplotlib==3.3.1
* timm==0.4.12
* torch==1.8.1
* editdistance==0.6.0
* opencv_python_headless==4.5.2.52
* numpy==1.19.5

