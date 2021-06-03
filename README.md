# 데이콘 - 공공 데이터 활용 온도 추정 AI 경진대회



## 시계열 데이터 / LSTM  / Transfer Learning



### https://dacon.io/competitions/official/235584/overview/description



## 데이터 설명

- 대전지역에서 측정한 실내외 19곳의 센서데이터와, 주변 지역의 기상청 공공데이터를 semi-비식별화하여 제공합니다.

- 센서는 온도를 측정하였습니다.

- 모든 데이터는 시간 순으로 정렬 되어 있으며 10분 단위 데이터 입니다.

- 예측 대상(target variable)은 Y18입니다.



## 데이터 전처리

- 기상청 데이터

  ```python
  temperature_name = ["X00","X07","X28","X31","X32"] #기온
  localpress_name  = ["X01","X06","X22","X27","X29"] #현지기압
  speed_name       = ["X02","X03","X18","X24","X26"] #풍속
  water_name       = ["X04","X10","X21","X36","X39"] #일일 누적강수량
  press_name       = ["X05","X08","X09","X23","X33"] #해면기압
  sun_name         = ["X11","X14","X16","X19","X34"] #일일 누적일사량
  humidity_name    = ["X12","X20","X30","X37","X38"] #습도
  direction_name   = ["X13","X15","X17","X25","X35"] #풍향
  ```

- 위의 기상청 데이터 중  각 지표별로 Y18 과 피어슨 상관계수가 가장 높은 피쳐 한 가지씩만을 사용

- ```python
  # 각 기온,기압,... 에서 가장 좋은 한가지 피쳐만을 사용
  req_x=['X00','X27','X24','X39','X33','X34','X12','X25']
  ```

- Y18 과 피어슨 상관계수값이 가장 높은 센서 Y16

- RNN 모델에 입력 할 수 있는 시계열 형태로 데이터 변환 

- 

## 모델

- Transfer Learning을 활용
- LSTM 모델을 사용
- Y18의 이전 데이터가 너무 적으므로, Y18과 가장 유사한 데이터인 Y16 데이터로 LSTM 모델을 Training
- 이후 LSTM Layer 는 고정 시킨 후, Y18 로 Dense Layer Fine Tunning 진행

- LSTM 모델 구축
  - LSTM Layer 1개에 Fully Connected Layer 2개 추가하여 생성
  - Optimizer 는 adam, loss 는 mse를 사용

```python
# lstm 모델 구축하기
simple_lstm_model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(128, input_shape=sequence.shape[-2:]),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(128, activation='relu'),
  
  tf.keras.layers.Dense(1)

])


simple_lstm_model.compile(optimizer='adam', loss='mse')
```



- Y16 validation 데이터 예측 결과

![predict_Y16](https://user-images.githubusercontent.com/28820900/120607812-807b6200-c48b-11eb-92c8-af9342ef3acc.PNG)



## Transfer Learning

- LSTM Layer는 고정시킨 후 , 더 적은 epoch 으로 DNN Layer에 대해서만 fine tunning 진행.
- fine tunning 은 실제 예측해야 하는 Y18 데이터로 진행

```python
# LSTM 레이어는 고정
simple_lstm_model.layers[0].trainable = False
# fine tuning 할 때 사용할 학습데이터 생성 (Y18)
finetune_X, finetune_y = convert_to_timeseries(pd.concat([X_train.tail(432), train['Y18'].tail(432)], axis = 1), interval=12)

# LSTM 레이어는 고정 시켜두고, DNN 레이어에 대해서 fine tuning 진행 (Transfer Learning)
finetune_history = simple_lstm_model.fit(
            finetune_X, finetune_y,
            epochs=1,
            batch_size=10,
            shuffle=False,
            verbose = 0)
```



- Y18 데이터 예측 결과

![predict_Y18](https://user-images.githubusercontent.com/28820900/120607832-84a77f80-c48b-11eb-8903-dae4de4d9b0d.PNG)

