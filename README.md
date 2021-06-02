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





## 

