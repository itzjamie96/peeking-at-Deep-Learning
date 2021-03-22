# Understanding basics of CNN

<br>

## 필수 패키지

- GPU를 이용해서 모델 학습할 때

  - CUDA
  - Cudnn

  ![image-20210323001511276](img/image-20210323001511276.png)

  

## cnn_training.py

> CNN 모델 학습

### TF 로그 필터링

```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

- 환경변수를 통해 TF의 수많은 로그 제어
- default = 0
- filter INFO logs = 1
- filter WARNING logs = 2
- filter ERROR logs = 3



## References

[TF 로그 필터링](https://yongyong-e.tistory.com/62)

[Limiting GPU memory growth](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/gpu.ipynb#scrollTo=ARrRhwqijPzN)

